#!/usr/bin/env python3
"""Log model-only weights for a stopped GXPO checkpoint to the existing W&B run."""

import sys
from pathlib import Path

import wandb


def main() -> None:
    checkpoint = Path(sys.argv[1]).resolve()
    expected_id = "4tfba63d"
    project = "gxpo-efficiency-final"
    run_name = "qwen25_math_1p5b_gxpo_k10_a05_b256_mb64_w50_tau3_v1_20260820"

    model_file = next(checkpoint.glob("actor/model_world_size_*_rank_0.pt"))
    hf_files = sorted((checkpoint / "actor/huggingface").glob("*"))
    weight_files = [model_file, *[path for path in hf_files if path.is_file()]]
    print(f"WEIGHT_FILES {len(weight_files)}", flush=True)
    print(
        f"WEIGHT_BYTES {sum(path.stat().st_size for path in weight_files)}",
        flush=True,
    )

    run = wandb.init(
        project=project,
        name=run_name,
        id=expected_id,
        resume="allow",
        job_type="weights-upload",
        settings=wandb.Settings(_disable_stats=True),
    )
    print(f"WANDB_RUN_ID {run.id}", flush=True)
    if run.id != expected_id:
        run.finish()
        raise RuntimeError(f"W&B resumed unexpected run id: {run.id}")

    run.log({
        "checkpoint/stopped_at_step": 175,
        "checkpoint/latest_complete_step": 175,
    })
    artifact = wandb.Artifact(
        name="gxpo_qwen25_math_1p5b_global_step_175_weights",
        type="model-weights",
        description=(
            "GXPO model weights captured after stopping at global step 175; "
            "optimizer and data-state files intentionally excluded."
        ),
        metadata={
            "method": "GXPO",
            "model": "Qwen2.5-Math-1.5B-Instruct",
            "global_step": 175,
            "k": 10,
            "alpha": 0.5,
            "batch_size": 256,
            "minibatch_size": 64,
            "warmup_steps": 50,
            "includes_optimizer_state": False,
            "includes_data_state": False,
        },
    )
    artifact.add_file(
        str(model_file),
        name="global_step_175/actor/" + model_file.name,
    )
    for path in hf_files:
        if path.is_file():
            artifact.add_file(
                str(path),
                name="global_step_175/actor/huggingface/" + path.name,
            )

    print("ARTIFACT_STAGED", flush=True)
    logged = run.log_artifact(artifact, aliases=["latest", "step-175"])
    print(f"ARTIFACT_LOGGED {logged.name}:{logged.version}", flush=True)
    run.summary.update({
        "checkpoint/latest_complete_step": 175,
        "checkpoint/weights_artifact": f"{logged.name}:{logged.version}",
        "checkpoint/stopped_by_user": True,
    })
    run.finish()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
