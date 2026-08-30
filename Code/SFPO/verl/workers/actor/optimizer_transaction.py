"""Local optimizer-state transactions for reversible probe updates.

The transaction deliberately operates on ``optimizer.state`` rather than
``state_dict()``.  The latter may materialize a global FSDP optimizer state,
while GXPO only needs to preserve the state owned by the current rank.
"""

from copy import deepcopy
from typing import Any

import torch


def _clone_value(value: Any) -> Any:
    """Clone optimizer state without retaining autograd history."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return deepcopy(value)


def _restore_value(current: Any, snapshot: Any) -> Any:
    """Restore a value in place where possible, preserving tensor references."""
    if isinstance(snapshot, torch.Tensor):
        if (isinstance(current, torch.Tensor) and current.shape == snapshot.shape
                and current.dtype == snapshot.dtype and current.device == snapshot.device):
            with torch.no_grad():
                current.copy_(snapshot)
            return current
        return snapshot.detach().clone()

    if isinstance(snapshot, dict):
        if not isinstance(current, dict):
            return _clone_value(snapshot)
        for key in list(current):
            if key not in snapshot:
                del current[key]
        for key, value in snapshot.items():
            current[key] = _restore_value(current.get(key), value)
        return current

    if isinstance(snapshot, list):
        if isinstance(current, list):
            current[:] = [_restore_value(item, value) for item, value in zip(current, snapshot)]
            if len(current) < len(snapshot):
                current.extend(_clone_value(value) for value in snapshot[len(current):])
            return current
        return _clone_value(snapshot)

    if isinstance(snapshot, tuple):
        return tuple(_restore_value(None, value) for value in snapshot)

    return deepcopy(snapshot)


class OptimizerStateTransaction:
    """Snapshot and restore one rank's optimizer state.

    Parameters remain owned by the caller because GXPO already has efficient
    parameter buffers.  ``restore`` is intentionally idempotent and restores
    tensor values in place whenever the optimizer kept the same allocation.
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        self._snapshot = {
            parameter: _clone_value(state)
            for parameter, state in optimizer.state.items()
        }
        self._closed = False

    def restore(self) -> None:
        """Restore the captured local state, including newly-created entries."""
        live_state = self.optimizer.state
        for parameter in list(live_state):
            if parameter not in self._snapshot:
                del live_state[parameter]

        for parameter, snapshot in self._snapshot.items():
            current = live_state.get(parameter)
            live_state[parameter] = _restore_value(current, snapshot)

    def commit(self) -> None:
        """Prevent a transaction context from restoring after a successful phase."""
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if not self._closed:
                self.restore()
        finally:
            self._closed = True
        return False


def snapshot_optimizer_state(optimizer: torch.optim.Optimizer) -> OptimizerStateTransaction:
    """Create a local optimizer-state transaction without global state gathering."""
    return OptimizerStateTransaction(optimizer)
