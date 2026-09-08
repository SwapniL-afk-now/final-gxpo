"""Offline top-K knowledge-distillation SFT trainer.

This module intentionally lives outside ``fsdp_sft_trainer``. Ordinary SFT
uses only response-only cross entropy; this trainer consumes the explicit
teacher cache columns produced by ``tools/kd_sft/build_teacher_topk.py``.
"""

import hydra

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
from verl.utils.dataset.kd_sft_dataset import KDSFTDataset
from verl.trainer.kd_sft_loss import KD_TOPK_CHUNK_TOKENS, compute_forward_kl_topk_chunked


def combine_hybrid_kd_loss(
    ce_loss: torch.Tensor,
    kd_loss: torch.Tensor,
    kd_alpha: float,
    kd_beta: float,
) -> torch.Tensor:
    """Hinton-style hybrid loss: alpha * hard-label CE + beta * (T-scaled) KL.

    Both inputs are per-token, already restricted to response positions
    (same [N] shape). Temperature scaling of ``kd_loss`` (the T^2 factor) is
    applied by the caller inside ``compute_forward_kl_topk_chunked`` -- this
    function only does the convex-ish combination, kept standalone so it is
    unit-testable without an FSDP model / GPU forward pass.
    """
    assert ce_loss.shape == kd_loss.shape, (ce_loss.shape, kd_loss.shape)
    assert kd_alpha >= 0.0 and kd_beta >= 0.0, (kd_alpha, kd_beta)
    return kd_alpha * ce_loss + kd_beta * kd_loss


def mean_per_sequence(values: torch.Tensor, sequence_ids: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Equal-weight mean of per-token values across sequences.

    Token-level averaging gives long solutions more influence than short ones.
    This helper first averages within each response and then averages the
    responses, while remaining differentiable with respect to ``values``.
    """
    if values.numel() == 0:
        return values.sum()
    sums = values.new_zeros(batch_size)
    sums.index_add_(0, sequence_ids, values)
    counts = torch.bincount(sequence_ids, minlength=batch_size).to(values.dtype)
    return (sums / counts.clamp_min(1)).mean()


class KDSFTTrainer(FSDPSFTTrainer):
    """FSDP SFT with a required, precomputed top-K teacher distribution."""

    def _build_dataloader(self):
        config = self.config

        def _as_plain(value):
            from omegaconf import ListConfig, DictConfig
            if isinstance(value, (ListConfig, DictConfig)):
                return OmegaConf.to_container(value, resolve=True)
            return value

        kd_kwargs = {
            'teacher_topk_log_probs_key': config.data.get(
                'teacher_topk_log_probs_key', 'teacher_topk_log_probs'),
            'teacher_topk_ids_key': config.data.get(
                'teacher_topk_ids_key', 'teacher_topk_ids'),
            'teacher_topk': config.data.get('teacher_topk', 32),
            'response_ids_key': config.data.get('response_ids_key', 'response_ids'),
            'include_eos': config.data.get('include_eos', True),
        }
        if not kd_kwargs['teacher_topk_log_probs_key'] or not kd_kwargs['teacher_topk_ids_key']:
            raise ValueError('KD-SFT requires teacher_topk_log_probs_key and teacher_topk_ids_key')

        dataset_kwargs = dict(
            tokenizer=self.tokenizer,
            prompt_key=config.data.prompt_key,
            prompt_dict_keys=config.data.get('prompt_dict_keys', None),
            response_key=config.data.response_key,
            response_dict_keys=config.data.get('response_dict_keys', None),
            max_length=config.data.max_length,
            truncation=config.data.truncation,
            **kd_kwargs,
        )
        self.train_dataset = KDSFTDataset(
            parquet_files=_as_plain(config.data.train_files), **dataset_kwargs)
        self.val_dataset = KDSFTDataset(
            parquet_files=_as_plain(config.data.val_files), **dataset_kwargs)

        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank,
            seed=config.trainer.get('seed', 0), drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset, batch_size=config.data.train_batch_size,
            sampler=self.train_sampler, num_workers=8, pin_memory=True, drop_last=True)
        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank,
            drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset, batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler, num_workers=8, pin_memory=True, drop_last=True)

    def _compute_loss_and_backward(self, batch: TensorDict, do_backward=True):
        """Compute cached top-K forward KL for one non-SP microbatch."""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        if use_sp:
            raise ValueError('offline KD-SFT does not support sequence parallelism')

        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        position_ids = batch['position_ids'].cuda()
        loss_mask_2d = batch.pop('loss_mask')[:, :-1].bool().cuda()
        # Teacher top-K rows are placed at input positions p..p+R-1 and
        # shifted by one below, matching the logits that predict them.
        kd_mask_2d = batch['kd_loss_mask'][:, 1:].bool().cuda()
        batch_size = input_ids.shape[0]
        sequence_grid = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand_as(loss_mask_2d)
        flat_mask = loss_mask_2d.reshape(-1)
        flat_kd_mask = kd_mask_2d.reshape(-1)

        # Hybrid loss weights: alpha on hard-label CE (teacher's realized
        # token, i.e. ordinary SFT), beta on the soft top-K forward-KL term.
        # Defaults (alpha=1, beta=0) reproduce the original pure-KD trainer
        # exactly -- this is opt-in, not a behavior change for existing runs.
        kd_alpha = float(self.config.data.get('kd_alpha', 1.0))
        kd_beta = float(self.config.data.get('kd_beta', 0.0))
        kd_temperature = float(self.config.data.get('kd_temperature', 1.0))
        if not bool(self.config.data.get('normalize_by_sequence', True)):
            raise ValueError('KD-SFT requires data.normalize_by_sequence=True to avoid length bias')

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.fsdp_model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=False)
            logits = output.logits
            shift_logits = logits[..., :-1, :]
            all_logits = shift_logits.reshape(-1, shift_logits.size(-1))
            response_logits = all_logits[flat_mask]
            response_sequence_ids = sequence_grid.reshape(-1)[flat_mask]

            if kd_beta > 0.0 and flat_kd_mask.any():
                kd_logits = all_logits[flat_kd_mask]
                kd_sequence_ids = sequence_grid.reshape(-1)[flat_kd_mask]
                flat_tlp = batch['teacher_topk_log_probs'].cuda()[:, 1:, :].reshape(
                    -1, batch['teacher_topk_log_probs'].shape[-1])[flat_kd_mask]
                flat_tid = batch['teacher_topk_ids'].cuda()[:, 1:, :].long().reshape(
                    -1, batch['teacher_topk_ids'].shape[-1])[flat_kd_mask]
                kd_out = compute_forward_kl_topk_chunked(
                    kd_logits, flat_tlp, flat_tid,
                    log_prob_min_clamp=self.config.data.get('kd_log_prob_min_clamp', None),
                    loss_max_clamp=self.config.data.get('kd_loss_max_clamp', None),
                    chunk_tokens=int(self.config.data.get(
                        'kd_chunk_tokens', KD_TOPK_CHUNK_TOKENS)),
                    temperature=kd_temperature,
                )
                kd_loss = kd_out['distillation_losses']
                kd_mean = mean_per_sequence(kd_loss, kd_sequence_ids, batch_size)
                del kd_logits, kd_sequence_ids, flat_tlp, flat_tid, kd_out, kd_loss
            else:
                kd_mean = response_logits.new_zeros(())

            if kd_alpha > 0.0:
                # Hard-label CE includes an explicit EOS token.
                flat_labels = input_ids[:, 1:].reshape(-1)[flat_mask]
                ce_loss = torch.nn.functional.cross_entropy(
                    response_logits.float(), flat_labels, reduction='none')
                ce_mean = mean_per_sequence(ce_loss, response_sequence_ids, batch_size)
                del ce_loss
            else:
                ce_mean = response_logits.new_zeros(())

            loss = combine_hybrid_kd_loss(ce_mean, kd_mean, kd_alpha, kd_beta)
            del output, logits, shift_logits, all_logits, response_logits

        # The normalized objective is already an equal-weight response mean;
        # do not divide it again by the number of tokens.
        if do_backward:
            loss.backward()
        return loss


# Keep the same base model/training contract as ordinary FSDP SFT, with dedicated KD defaults, but instantiate
# the dedicated trainer so a cache cannot accidentally enter ordinary SFT.
from torch.distributed.device_mesh import init_device_mesh
from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='kd_sft_trainer', version_base=None)
def main(config):
    _, _, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(
        device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type='cuda', mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=('dp', 'sp'))
    trainer = KDSFTTrainer(
        config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    trainer.fit()


if __name__ == '__main__':
    main()
