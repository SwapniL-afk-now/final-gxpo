# KodCode reward: function-based problems graded by their pytest suite.
# The model writes a module; KodCode's `test` string does `from solution import <fn>` and defines
# test_* functions with asserts. We write both to a temp dir, run pytest once, and return the
# fraction of test functions that pass (dense reward). Parsing/syntax/exception/timeout -> 0.0.
import os
import re
import subprocess
import tempfile

# Last fenced block, any language tag; fall back to post-fence text, then raw. Grade the final
# answer, not the fence syntax (mirrors prime_code.extract_code).
_FENCE_RE = re.compile(r"```[ \t]*[a-zA-Z0-9_+\-]*[ \t]*\r?\n?(.*?)```", re.DOTALL)


def extract_code(completion):
    blocks = _FENCE_RE.findall(completion)
    if blocks:
        return blocks[-1].strip("\n")
    if "```" in completion:
        return completion.rsplit("```", 1)[-1].strip("\n")
    return completion


def _run_pytest(solution_code, test_code, timeout):
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "solution.py"), "w") as f:
            f.write(solution_code)
        with open(os.path.join(td, "test_solution.py"), "w") as f:
            f.write(test_code)
        try:
            r = subprocess.run(
                ["python", "-m", "pytest", "test_solution.py", "-q", "--no-header",
                 "-p", "no:cacheprovider", "-o", "addopts="],
                cwd=td, capture_output=True, text=True, timeout=timeout,
            )
        except (subprocess.TimeoutExpired, Exception):
            return 0.0
        out = r.stdout + r.stderr
        m = re.search(r"(\d+) passed", out)
        f_ = re.search(r"(\d+) failed", out)
        e_ = re.search(r"(\d+) error", out)
        passed = int(m.group(1)) if m else 0
        failed = (int(f_.group(1)) if f_ else 0) + (int(e_.group(1)) if e_ else 0)
        total = passed + failed
        # collection error / syntax error in solution -> no tests ran -> 0
        return passed / total if total else 0.0


def compute_score(completion, test_code, continuous=True, timeout=10):
    """Dense KodCode reward in [0,1] = fraction of pytest tests passed. `test_code` is the KodCode
    `test` string (ground_truth). `continuous` accepted for dispatch symmetry (always dense here)."""
    solution = extract_code(completion)
    if not solution.strip():
        return 0.0
    try:
        return float(_run_pytest(solution, test_code, timeout))
    except Exception:
        return 0.0


def _selfcheck():
    """Runnable check: reference solution -> 1.0, wrong stub -> 0.0.
    Run: python -c "from verl.utils.reward_score.kodcode import _selfcheck as f; f()" """
    from datasets import load_dataset
    d = load_dataset("KodCode/KodCode-Light-RL-10K", split="train")
    ex = d[0]
    good = compute_score(f"```python\n{ex['solution']}\n```", ex["test"])
    bad = compute_score("```python\ndef nope():\n    return None\n```", ex["test"])
    assert good == 1.0, good
    assert bad == 0.0, bad
    print("kodcode reward selfcheck OK (ref=1.0, wrong=0.0)")


if __name__ == "__main__":
    _selfcheck()
