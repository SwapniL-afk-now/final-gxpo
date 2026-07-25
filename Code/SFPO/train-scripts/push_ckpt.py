#!/usr/bin/env python
"""Merge a single-GPU (NO_SHARD) verl actor checkpoint to HF format and upload.

verl's scripts/model_merger.py assumes sharded DTensors; our fsdp_size=1 runs save a
plain state_dict, so we just load config+weights, save_pretrained, and push the folder.
Usage: python train-scripts/push_ckpt.py <actor_dir> [<hf_repo_id>] [--private]
Omit the repo id to convert in place without uploading.
"""
import argparse, os, torch
from transformers import AutoConfig, AutoModelForCausalLM
from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("actor_dir")            # runs/<run>/global_step_X/actor
ap.add_argument("repo_id", nargs="?")   # e.g. ismamNur/grpo-mathl35-amc23-seed42; omit to only convert locally
ap.add_argument("--private", action="store_true")
args = ap.parse_args()

hf_dir = os.path.join(args.actor_dir, "huggingface")
[pt] = [f for f in os.listdir(args.actor_dir) if f.startswith("model_world_size_") and f.endswith("rank_0.pt")]
sd = torch.load(os.path.join(args.actor_dir, pt), map_location="cpu")
sd = {k: v.to(torch.bfloat16) for k, v in sd.items()}

config = AutoConfig.from_pretrained(hf_dir)
with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
model.to_empty(device="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not unexpected, f"unexpected keys: {unexpected[:5]}"
assert not missing, f"missing keys: {missing[:5]}"          # weights must fully populate the model

model.save_pretrained(hf_dir, safe_serialization=True)      # writes model.safetensors alongside config/tokenizer
print(f"saved HF model -> {hf_dir}")

if not args.repo_id:
    raise SystemExit(0)                                     # convert-only mode

api = HfApi()
api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
api.upload_folder(folder_path=hf_dir, repo_id=args.repo_id, repo_type="model")
print(f"uploaded -> https://huggingface.co/{args.repo_id} (private={args.private})")
