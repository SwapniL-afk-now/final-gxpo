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
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.fsdp_model(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, use_cache=False)
            logits = output.logits
            shift_logits = logits[..., :-1, :]
            flat_mask = loss_mask.bool()
            flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))[flat_mask]
            flat_tlp = batch['teacher_topk_log_probs'].cuda()[:, 1:, :].reshape(
                -1, batch['teacher_topk_log_probs'].shape[-1])[flat_mask]
            flat_tid = batch['teacher_topk_ids'].cuda()[:, 1:, :].long().reshape(
                -1, batch['teacher_topk_ids'].shape[-1])[flat_mask]
            kd_out = compute_forward_kl_topk_chunked(
                flat_logits, flat_tlp, flat_tid,
                log_prob_min_clamp=self.config.data.get('kd_log_prob_min_clamp', None),
                loss_max_clamp=self.config.data.get('kd_loss_max_clamp', None),
                chunk_tokens=int(self.config.data.get(
                    'kd_chunk_tokens', KD_TOPK_CHUNK_TOKENS)),
            )
            loss = kd_out['distillation_losses']
            del output, logits, shift_logits, flat_logits, flat_tlp, flat_tid, kd_out

        valid_token_this_rank = torch.sum(loss_mask)
        if self.config.data.balance_dp_token:
            torch.distributed.all_reduce(valid_token_this_rank)
            dp_size = torch.distributed.get_world_size()
        else:
            dp_size = 1
        loss = torch.sum(loss) / valid_token_this_rank * dp_size
        if do_backward:
            loss.backward()
        return loss


# Keep the same Hydra/config contract as ordinary FSDP SFT, but instantiate
# the dedicated trainer so a cache cannot accidentally enter ordinary SFT.
from torch.distributed.device_mesh import init_device_mesh
from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
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
