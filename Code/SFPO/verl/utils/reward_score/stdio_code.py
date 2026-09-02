# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Stdin/stdout code-execution reward (LiveCodeBench's `testtype: stdin` problems).

This fork's `prime_code` module depends on `pyext`, which is incompatible with
Python 3.12 (it calls the removed `inspect.getargspec`). Rather than depend on a
sandbox_fusion service or a broken local module, this implements a minimal,
self-contained subprocess runner: the candidate program is executed as a script,
fed each test's stdin, and its stdout is compared (whitespace-normalized) against
the expected output.
"""

from __future__ import annotations

import json
import math
import os
import re
import resource
import signal
import subprocess
import sys
import time

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_MEMORY_LIMIT_MB = max(128, int(os.environ.get("VERIFIER_MEMORY_MB", "2048")))


def _set_candidate_limits(timeout: float) -> None:
    """Apply fail-closed CPU/address-space limits in the candidate child."""
    try:
        cpu_limit = max(1, math.ceil(timeout) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
        memory_limit = _MEMORY_LIMIT_MB * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
    except (OSError, ValueError):
        # The wall-clock timeout and process-group kill remain the fallback on
        # platforms where one of these limits is unavailable.
        pass


def _extract_code(solution_str: str) -> str:
    matches = _CODE_FENCE_RE.findall(solution_str)
    return matches[-1] if matches else solution_str


def _normalize(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def compute_score(
    solution_str: str,
    ground_truth: str | dict,
    extra_info: dict | None = None,
    timeout: float = 6.0,
    continuous: bool = True,
    stop_on_failure: bool = False,
) -> dict:
    """Score a completion against stdin/stdout test cases.

    Each test case runs in a fresh subprocess. This function is called inside a
    bounded verifier worker, so the former extra Manager/process wrapper is
    unnecessary and would multiply process-launch overhead. ``stop_on_failure``
    is intended for offline filtering, where a candidate must pass every test;
    it leaves the default continuous reward behavior unchanged.
    """
    spec = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    inputs = spec["inputs"]
    outputs = spec["outputs"]
    code = _extract_code(solution_str)

    passed = 0
    deadline = time.monotonic() + timeout
    for stdin, expected in zip(inputs, outputs, strict=True):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                preexec_fn=lambda: _set_candidate_limits(timeout),
            )
            try:
                stdout, _ = proc.communicate(input=stdin, timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
                if stop_on_failure:
                    break
                continue
            if proc.returncode == 0 and _normalize(stdout) == _normalize(expected):
                passed += 1
            elif stop_on_failure:
                break
        except Exception:
            if stop_on_failure:
                break
            continue

    total = len(inputs) or 1
    frac = passed / total
    score = frac if continuous else (1.0 if passed == total else -1.0)
    return {"score": score, "acc": passed == total, "pass_rate": frac}
