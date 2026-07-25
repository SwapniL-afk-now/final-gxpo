# Copyright 2024 PRIME team and/or its affiliates
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

from .utils import check_correctness as apps_check_correctness
import json
import os
import re
import traceback

# Match a fenced block with any (or no) language tag: ```python, ```py, ```Python, or plain ```.
_FENCE_RE = re.compile(r"```[ \t]*[a-zA-Z0-9_+\-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


def extract_code(completion):
    """Grade the final answer, not the fence syntax. Take the LAST fenced code block
    regardless of its language tag; if a fence was opened but not closed, take everything
    after the last opening fence; if there is no fence at all, treat the whole completion as
    code. The old `split('```python')[-1].split('```')[0]` scored 0 for any block not tagged
    exactly ```python (grabbing the prose before it) -> inflated 'format_error_ratio'."""
    blocks = _FENCE_RE.findall(completion)
    if blocks:
        return blocks[-1].strip("\n")
    if "```" in completion:
        return completion.rsplit("```", 1)[-1].strip("\n")
    return completion

# Partial-credit loop cap: for a failing completion, how many test cases to probe
# individually (each spawns its own process). The original hardcoded 10 dominated
# adv time; REWARD_CONTINUOUS_MAX lets code runs trade a little reward granularity
# for a big speedup. Default 10 preserves prior behavior.
_CONTINUOUS_MAX = max(1, int(os.environ.get("REWARD_CONTINUOUS_MAX", "10")))


def compute_score(completion, test_cases, continuous=False):
    # try to get code solution from completion. if the completion is pure code, this will not take effect.
    solution = extract_code(completion)
    try:
        try:
            if not isinstance(test_cases, dict):
                test_cases = json.loads(test_cases)
        except Exception as e:
            print(f"Error:{e}")

        # Complete check on all in-out pairs first. If there is no failure, per-sample test can be skipped.
        try:
            res, metadata = apps_check_correctness(in_outs=test_cases, generation=solution, timeout=5, debug=False)
            metadata = dict(enumerate(metadata))[0]
            success = all(map(lambda x: x == True, res))
            if success:
                return success, metadata
        except Exception as e:
            pass

        test_cases_list = []
        inputs = test_cases["inputs"]
        outputs = test_cases["outputs"]
        for i in range(len(inputs)):
            test_cases_list.append({"inputs": [inputs[i]], "outputs": [outputs[i]]})

        if continuous:
            # per sample test: if continuous score is needed, test first 10 samples regardless of failures
            # do not test all samples cuz some problems have enormous test cases
            metadata_list = []
            res_list = []
            for test_case_id, test_case in enumerate(test_cases_list):
                res, metadata = apps_check_correctness(in_outs=test_case, generation=solution, timeout=5, debug=False)
                try:
                    metadata = dict(enumerate(metadata))[0]  # metadata can be empty occasionally
                except Exception as e:
                    metadata = {}
                metadata["test_case"] = {}
                metadata["test_case"]["input"] = str(test_case["inputs"][0])
                metadata["test_case"]["output"] = str(test_case["outputs"][0])
                metadata["test_case"]["res"] = str(res)
                metadata_list.append(metadata)
                res_list.extend(res)

                if test_case_id >= _CONTINUOUS_MAX - 1:
                    break
            res_count = len(res_list) if len(res_list) > 0 else 1
            success = sum(map(lambda x: x == True, res_list)) / res_count
    except Exception as e:
        traceback.print_exc(10)
        success = False
        metadata_list = None
    return success, metadata_list


def _selfcheck_extract():
    """Runnable check for extract_code: every fenced variant must yield the code, not the prose.
    Run: python -c "from verl.utils.reward_score.prime_code import _selfcheck_extract as f; f()" """
    C = "n=int(input())\nprint(n*2)"
    cases = {
        "python": f"sol:\n```python\n{C}\n```<|im_end|>",
        "py": f"```py\n{C}\n```",
        "Python": f"```Python\n{C}\n```",
        "plain": f"text\n```\n{C}\n```",
        "purecode": C,
        "trailing": f"```python\n{C}\n```\ndone!",
        "twoblocks": f"```python\nprint('draft')\n```\nfinal\n```python\n{C}\n```",
    }
    for name, comp in cases.items():
        got = extract_code(comp).strip()
        assert got == C, f"{name}: {got!r}"
    print("prime_code.extract_code selfcheck: 7/7 OK")


if __name__ == "__main__":
    _selfcheck_extract()
