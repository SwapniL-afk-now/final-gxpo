"""Small dependency-free helpers shared by the FSDP SFT trainer and tests."""


def resolve_total_training_steps(steps_per_epoch, total_epochs, configured_total_steps=None):
    """Return the one step horizon used by both the scheduler and fit loop.

    configured_total_steps is an optional hard cap. A cap larger than the
    available epoch budget is harmless and resolves to the natural budget.
    """
    steps_per_epoch = int(steps_per_epoch)
    total_epochs = int(total_epochs)
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be positive, got {steps_per_epoch}")
    if total_epochs <= 0:
        raise ValueError(f"total_epochs must be positive, got {total_epochs}")

    natural_steps = steps_per_epoch * total_epochs
    if configured_total_steps is None:
        return natural_steps

    configured_total_steps = int(configured_total_steps)
    if configured_total_steps <= 0:
        raise ValueError(
            f"trainer.total_training_steps must be positive when set, got {configured_total_steps}"
        )
    return min(natural_steps, configured_total_steps)
