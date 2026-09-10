"""Every rank must produce the same micro-batch COUNT, or FSDP deadlocks.

Under data.use_dynamic_bsz the count depends on the rank's own sequence lengths.
pad_groups_to is what re-aligns them; without it rank0 sat in a 1-element
ALLREDUCE while rank1 was in _ALLGATHER_BASE until the 600s NCCL watchdog fired.
"""
import importlib.util
import pathlib
import sys

sys.modules.setdefault('verl', type(sys)('verl'))
_src = pathlib.Path(__file__).resolve().parents[2] / 'verl/trainer/fsdp_sft_trainer.py'
# Import just the helper without dragging in torch/FSDP module-level imports.
_ns = {}
_text = _src.read_text()
_start = _text.index('def pad_groups_to')
_end = _text.index('class FSDPSFTTrainer')
exec(compile(_text[_start:_end], str(_src), 'exec'), _ns)
pad_groups_to = _ns['pad_groups_to']


def _split_by_budget(lengths, budget):
    """The grouping logic from _split_micro_batches, for building rank inputs."""
    groups, current, current_max = [], [], 0
    for index in sorted(range(len(lengths)), key=lambda i: lengths[i]):
        widened = max(current_max, lengths[index])
        if current and (len(current) + 1) * widened > budget:
            groups.append(current)
            current, current_max = [index], lengths[index]
        else:
            current.append(index)
            current_max = widened
    if current:
        groups.append(current)
    return groups


def test_ranks_agree_on_count():
    budget = 12288
    # The real failure shape: same row count, very different length mixes.
    rank0 = [10425, 9800, 8100, 7500, 3200, 3100, 900, 800]   # long rows -> many groups
    rank1 = [900, 850, 800, 780, 760, 740, 720, 700]          # short rows -> one group
    g0, g1 = _split_by_budget(rank0, budget), _split_by_budget(rank1, budget)
    assert len(g0) != len(g1), 'precondition: this input must desync without the fix'

    target = max(len(g0), len(g1))
    p0, p1 = pad_groups_to(g0, target), pad_groups_to(g1, target)
    assert len(p0) == len(p1) == target

    # No rows invented or lost, and the budget still holds on every group.
    for original, padded, lengths in ((g0, p0, rank0), (g1, p1, rank1)):
        assert sorted(i for g in padded for i in g) == sorted(i for g in original for i in g)
        for group in padded:
            assert len(group) * max(lengths[i] for i in group) <= budget


def test_noop_when_already_aligned():
    groups = [[0, 1], [2, 3]]
    assert pad_groups_to(groups, 2) == groups


def test_stops_when_all_singletons():
    # target above the row count is unreachable; must not spin forever.
    assert len(pad_groups_to([[0], [1]], 5)) == 2


if __name__ == '__main__':
    test_ranks_agree_on_count()
    test_noop_when_already_aligned()
    test_stops_when_all_singletons()
    print('OK: micro-batch counts align across ranks')
