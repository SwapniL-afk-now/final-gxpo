"""Muon optimizer (Moonlight / Kimi reference implementation).

Adapted from https://github.com/MoonshotAI/Moonlight (examples/toy_train.py), which is
itself a modified version of https://github.com/KellerJordan/Muon/blob/master/muon.py

Requires 2-D parameters. Under distributed FSDP1, the actor is built with
`use_orig_params=True`; sharded local views are flattened to 1-D, so the gather-scatter
backend reconstructs each original matrix before Newton–Schulz and writes back its local
slice. See `build_fsdp_muon_registry` and `fsdp_workers._build_model_optimizer`.
"""
import math
import hashlib
import json
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


MUON_BACKEND_VERSION = 'gather_scatter_v1'


@dataclass(frozen=True)
class FSDPMuonParamInfo:
    """Stable mapping from one original parameter to its FSDP local shard."""

    param: torch.nn.Parameter
    name: str
    flat_param: object
    shape: tuple
    contiguous: bool
    process_group: object
    group_world_size: int
    global_numel: int
    global_offset: int
    local_start: int
    local_numel: int


def _product(shape):
    return math.prod(int(x) for x in shape)


def _is_embedding_or_head(name):
    """Return whether a parameter name denotes an embedding or output head."""
    lowered = name.lower()
    return any(token in lowered for token in (
        "embed_tokens", "embedding", "word_embeddings", "tok_embeddings",
        ".wte", "lm_head", "output_head", "output_projection", "output_layer"))


def build_fsdp_muon_registry(fsdp_model):
    """Build the original-parameter -> local-shard map used by distributed Muon.

    FSDP1 exposes original parameters as flattened local views when sharded. The
    owning FlatParameter retains the original shapes and the local intersection of
    each parameter with the rank's shard. We keep access to that version-sensitive
    metadata in this one function and validate it eagerly so a broken mapping cannot
    silently become an AdamW run.
    """
    if not dist.is_initialized():
        raise RuntimeError('distributed Muon requires an initialized process group')

    handles = list(getattr(fsdp_model, '_all_handles', ()))
    if not handles:
        handle = getattr(fsdp_model, '_handle', None)
        if handle is not None:
            handles = [handle]
    if not handles:
        raise RuntimeError('distributed Muon could not find any FSDP handles')

    registry = {}
    for handle in handles:
        flat = getattr(handle, 'flat_param', None)
        required = ('_params', '_fqns', '_shapes', '_contiguities',
                    '_numels_with_padding', '_is_padding_mask', '_shard_param_infos')
        if flat is None or any(not hasattr(flat, key) for key in required):
            raise RuntimeError(
                'distributed Muon requires FSDP1 original-parameter metadata '
                f'({required})')

        params = flat._params
        if params is None:
            raise RuntimeError(
                'distributed Muon requires use_orig_params=True; FSDP returned no original parameters')
        fqns = flat._fqns
        shapes = flat._shapes
        contiguities = flat._contiguities
        shard_infos = flat._shard_param_infos
        numels_with_padding = flat._numels_with_padding
        is_padding_mask = flat._is_padding_mask
        if len(numels_with_padding) != len(is_padding_mask):
            raise RuntimeError(f'FSDP Muon padding metadata length mismatch for {flat!r}')
        numels = []
        global_offsets = []
        flat_offset = 0
        for numel, is_padding in zip(numels_with_padding, is_padding_mask):
            numel = int(numel)
            if not is_padding:
                numels.append(numel)
                global_offsets.append(flat_offset)
            flat_offset += numel
        count = len(params)
        if not (len(numels) == len(fqns) == len(shapes) == len(contiguities) == len(shard_infos) == count):
            raise RuntimeError(f'FSDP Muon metadata length mismatch for {flat!r}')

        process_group = getattr(handle, 'process_group', None)
        if process_group is None:
            raise RuntimeError('FSDP Muon handle has no process group')
        group_world_size = dist.get_world_size(process_group)

        for index, (param, name, shape, contiguous, shard_info) in enumerate(
                zip(params, fqns, shapes, contiguities, shard_infos)):
            global_numel = _product(shape)
            padded_numel = numels[index]
            if padded_numel < global_numel:
                raise RuntimeError(f'invalid FSDP Muon metadata for {name}: padded numel is too small')

            if shard_info.in_shard:
                local_start = int(shard_info.intra_param_start_idx)
                local_numel = int(shard_info.numel_in_shard)
            else:
                local_start = 0
                local_numel = 0

            if local_numel and local_start + local_numel > global_numel:
                raise RuntimeError(f'invalid local shard range for FSDP Muon parameter {name}')
            if param in registry:
                raise RuntimeError(f'duplicate/shared FSDP Muon parameter is unsupported: {name}')

            registry[param] = FSDPMuonParamInfo(
                param=param,
                name=name,
                flat_param=flat,
                shape=tuple(int(x) for x in shape),
                contiguous=bool(contiguous),
                process_group=process_group,
                group_world_size=group_world_size,
                global_numel=global_numel,
                global_offset=global_offsets[index],
                local_start=local_start,
                local_numel=local_numel,
            )

    if not registry:
        raise RuntimeError('distributed Muon found no original FSDP parameters')
    return registry


# ponytail: eager NS5 (upstream decorates this with @torch.compile). ~10 distinct 2-D
# shapes in a Qwen2.5-1.5B exceed dynamo's default cache_size_limit=8 -> recompile churn,
# and 15 bf16 matmuls/param is noise against a rollout-dominated 75-105s step. Re-add
# torch.compile plus a raised cache_size_limit if profiling ever says otherwise.
def zeropower_via_newtonschulz5(G, steps):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Muon with an FSDP1 gather-compute-scatter backend.

    In the distributed path the optimizer sees FSDP's flattened local original-
    parameter views.  It all-gathers each matrix's gradient and local momentum,
    applies the dense Muon update to the reconstructed global matrix, then writes
    only the rank-local slice back.  Optimizer state remains sharded locally.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        fsdp_model=None,
        fsdp_registry=None,
        parameter_signature=None,
    ):
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )
        muon_params = list(muon_params or [])
        adamw_params = list(adamw_params or [])
        params = muon_params + adamw_params
        if len({id(p) for p in params}) != len(params):
            raise ValueError('Muon optimizer received duplicate parameters')
        super().__init__(params, defaults)

        self.fsdp_model = fsdp_model
        self.fsdp_registry = fsdp_registry or {}
        self.distributed = bool(self.fsdp_registry and dist.is_initialized() and dist.get_world_size() > 1)
        self.backend = 'gather_scatter' if self.distributed else 'dense'
        self.backend_version = MUON_BACKEND_VERSION
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.fsdp_size = max((info.group_world_size for info in self.fsdp_registry.values()), default=1)
        self.muon_parameter_count = len(muon_params)
        self.adamw_parameter_count = len(adamw_params)
        self.muon_parameter_numel = sum(
            self.fsdp_registry[p].global_numel if p in self.fsdp_registry else p.numel()
            for p in muon_params)
        self.adamw_parameter_numel = sum(
            self.fsdp_registry[p].global_numel if p in self.fsdp_registry else p.numel()
            for p in adamw_params)
        self.last_diagnostics = {}
        self.parameter_signature = parameter_signature

        for p in muon_params:
            if not self.distributed and p.ndim != 2:
                raise ValueError(f'dense Muon parameter must be 2-D, got {tuple(p.shape)}')
            self.state[p]['use_muon'] = True
        for p in adamw_params:
            self.state[p]['use_muon'] = False

        if self.distributed:
            for p, info in self.fsdp_registry.items():
                if p.requires_grad and info.group_world_size > 1:
                    local_numel = info.local_numel
                    if p.data.numel() != local_numel:
                        raise RuntimeError(
                            f'FSDP Muon local view mismatch for {info.name}: '
                            f'parameter has {p.data.numel()} elements, registry has {local_numel}')
                    if not info.contiguous:
                        raise RuntimeError(f'FSDP Muon requires contiguous parameter {info.name}')

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

    @staticmethod
    def _is_present(parameter):
        return parameter.grad is not None

    def _local_tensor(self, tensor, info, dtype=None):
        device = info.param.device
        if tensor is None:
            return torch.zeros(info.local_numel, device=device, dtype=dtype or info.param.dtype)
        local = tensor.detach().reshape(-1)
        if local.numel() != info.local_numel:
            raise RuntimeError(
                f'FSDP Muon local tensor mismatch for {info.name}: '
                f'got {local.numel()}, expected {info.local_numel}')
        if dtype is not None and local.dtype != dtype:
            local = local.to(dtype)
        return local.contiguous()

    def _all_gather_full(self, values, info):
        """Gather variable-length local slices and place them by intra-param offset."""
        if info.group_world_size == 1:
            return [v.reshape(info.shape) for v in values]

        device = values[0].device
        group = info.process_group
        local_meta = torch.tensor(
            [info.local_start, info.local_numel], device=device, dtype=torch.int64)
        metas = [torch.empty_like(local_meta) for _ in range(info.group_world_size)]
        dist.all_gather(metas, local_meta, group=group)
        starts = [int(meta[0].item()) for meta in metas]
        lengths = [int(meta[1].item()) for meta in metas]
        max_len = max(lengths, default=0)
        n_values = len(values)
        payload = torch.zeros(n_values * max_len, device=device, dtype=values[0].dtype)
        for index, value in enumerate(values):
            if value.numel():
                payload[index * max_len:index * max_len + value.numel()].copy_(value)
        gathered = [torch.empty_like(payload) for _ in range(info.group_world_size)]
        dist.all_gather(gathered, payload, group=group)

        full_values = [torch.zeros(info.global_numel, device=device, dtype=values[0].dtype)
                       for _ in range(n_values)]
        for rank, payload_rank in enumerate(gathered):
            start = starts[rank]
            length = lengths[rank]
            if length:
                for index in range(n_values):
                    source = payload_rank[index * max_len:index * max_len + length]
                    full_values[index][start:start + length].copy_(source)
        return [full.reshape(info.shape) for full in full_values]

    def _ensure_distributed_state(self, parameter, info, use_muon):
        state = self.state[parameter]
        if info.local_numel == 0:
            return
        device = parameter.device
        dtype = parameter.dtype
        if use_muon:
            current = state.get('momentum_buffer')
            if current is None:
                state['momentum_buffer'] = torch.zeros(info.local_numel, device=device, dtype=dtype)
            elif current.device != device or current.numel() != info.local_numel:
                if current.device.type == 'cpu':
                    raise RuntimeError('distributed Muon does not support CPU optimizer offload')
                state['momentum_buffer'] = current.to(device=device, dtype=dtype).reshape(-1)
        else:
            if state.get('step') is None:
                state['step'] = 0
            for key in ('moment1', 'moment2'):
                current = state.get(key)
                if current is None:
                    state[key] = torch.zeros(info.local_numel, device=device, dtype=dtype)
                elif current.device != device or current.numel() != info.local_numel:
                    if current.device.type == 'cpu':
                        raise RuntimeError('distributed Muon does not support CPU optimizer offload')
                    state[key] = current.to(device=device, dtype=dtype).reshape(-1)

    def _step_dense(self):
        update_sq = None
        momentum_sq = None
        for group in self.param_groups:
            lr = group['lr']
            wd = group['wd']
            momentum = group['momentum']
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue
                g = p.grad
                if state['use_muon']:
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    direction = g.add(buf, alpha=momentum) if group['nesterov'] else buf
                    update = zeropower_via_newtonschulz5(direction, steps=group['ns_steps'])
                    p.data.mul_(1 - lr * wd)
                    p.data.add_(update, alpha=-self.adjust_lr_for_muon(lr, p.shape))
                    if update_sq is None:
                        update_sq = update.float().square().sum()
                        momentum_sq = buf.float().square().sum()
                    else:
                        update_sq += update.float().square().sum()
                        momentum_sq += buf.float().square().sum()
                else:
                    if 'step' not in state:
                        state['step'] = 0
                        state['moment1'] = torch.zeros_like(g)
                        state['moment2'] = torch.zeros_like(g)
                    state['step'] += 1
                    step = state['step']
                    beta1, beta2 = group['adamw_betas']
                    buf1, buf2 = state['moment1'], state['moment2']
                    buf1.lerp_(g, 1 - beta1)
                    buf2.lerp_(g.square(), 1 - beta2)
                    normalized = buf1 / (group['adamw_eps'] + buf2.sqrt())
                    correction = (1 - beta1**step) / (1 - beta2**step)**0.5
                    p.data.mul_(1 - lr * group['wd'])
                    p.data.add_(normalized, alpha=-lr / correction)

        if update_sq is None:
            device = self.param_groups[0]['params'][0].device
            update_sq = torch.zeros((), device=device, dtype=torch.float32)
            momentum_sq = torch.zeros((), device=device, dtype=torch.float32)
        self.last_diagnostics = {
            'optimizer/muon_update_norm': float(update_sq.sqrt().item()),
            'optimizer/muon_momentum_norm': float(momentum_sq.sqrt().item()),
            'optimizer/muon_gather_time': 0.0,
            'optimizer/muon_newton_schulz_time': 0.0,
            'optimizer/muon_scatter_time': 0.0,
            'optimizer/muon_collective_bytes': 0.0,
        }

    def _step_distributed(self):
        gather_s = 0.0
        ns_s = 0.0
        scatter_s = 0.0
        collective_bytes = 0
        update_sq = None
        momentum_sq = None

        for group in self.param_groups:
            lr = group['lr']
            wd = group['wd']
            momentum = group['momentum']
            for parameter in group['params']:
                info = self.fsdp_registry.get(parameter)
                if info is None:
                    raise RuntimeError('distributed Muon parameter is missing from the FSDP registry')
                state = self.state[parameter]
                use_muon = bool(state['use_muon'])
                if use_muon:
                    self._ensure_distributed_state(parameter, info, use_muon=True)
                    present = torch.tensor(
                        [1 if self._is_present(parameter) else 0],
                        device=parameter.device, dtype=torch.int32)
                    dist.all_reduce(present, op=dist.ReduceOp.MAX, group=info.process_group)
                    if present.item() == 0:
                        continue

                    local_grad = self._local_tensor(parameter.grad, info, dtype=parameter.dtype)
                    local_momentum = self._local_tensor(
                        state['momentum_buffer'], info, dtype=parameter.dtype)
                    gather_start = time.perf_counter()
                    full_grad, full_momentum = self._all_gather_full(
                        [local_grad, local_momentum], info)
                    gather_s += time.perf_counter() - gather_start
                    collective_bytes += info.group_world_size * (
                        2 * max(info.local_numel, 1) * parameter.element_size()
                        + 16)  # two int64 metadata values per gathered slice
                    collective_bytes += info.group_world_size * 4  # presence all-reduce

                    ns_start = time.perf_counter()
                    full_momentum.mul_(momentum).add_(full_grad)
                    direction = (full_grad.add(full_momentum, alpha=momentum)
                                 if group['nesterov'] else full_momentum)
                    update = zeropower_via_newtonschulz5(direction, steps=group['ns_steps'])
                    ns_s += time.perf_counter() - ns_start

                    local_data = parameter.data.reshape(-1)
                    if local_data.numel() != info.local_numel:
                        raise RuntimeError(f'FSDP Muon parameter storage changed for {info.name}')
                    local_update = update.reshape(-1)[info.local_start:info.local_start + info.local_numel]
                    local_momentum_new = full_momentum.reshape(-1)[
                        info.local_start:info.local_start + info.local_numel]
                    scatter_start = time.perf_counter()
                    local_data.mul_(1 - lr * wd)
                    local_data.add_(local_update, alpha=-self.adjust_lr_for_muon(lr, info.shape))
                    state['momentum_buffer'] = local_momentum_new.detach().clone()
                    scatter_s += time.perf_counter() - scatter_start

                    if update_sq is None:
                        update_sq = update.float().square().sum()
                        momentum_sq = full_momentum.float().square().sum()
                    else:
                        update_sq += update.float().square().sum()
                        momentum_sq += full_momentum.float().square().sum()
                    del full_grad, full_momentum, direction, update, local_update, local_momentum_new
                else:
                    if parameter.grad is None:
                        continue
                    self._ensure_distributed_state(parameter, info, use_muon=False)
                    grad = self._local_tensor(parameter.grad, info, dtype=parameter.dtype)
                    state['step'] += 1
                    step = state['step']
                    beta1, beta2 = group['adamw_betas']
                    buf1, buf2 = state['moment1'], state['moment2']
                    buf1.lerp_(grad, 1 - beta1)
                    buf2.lerp_(grad.square(), 1 - beta2)
                    normalized = buf1 / (group['adamw_eps'] + buf2.sqrt())
                    correction = (1 - beta1**step) / (1 - beta2**step)**0.5
                    parameter.data.reshape(-1).mul_(1 - lr * wd)
                    parameter.data.reshape(-1).add_(normalized, alpha=-lr / correction)

        if update_sq is None:
            device = next(iter(self.state)).device
            update_sq = torch.zeros((), device=device, dtype=torch.float32)
            momentum_sq = torch.zeros((), device=device, dtype=torch.float32)
        self.last_diagnostics = {
            'optimizer/muon_update_norm': float(update_sq.sqrt().item()),
            'optimizer/muon_momentum_norm': float(momentum_sq.sqrt().item()),
            'optimizer/muon_gather_time': gather_s,
            'optimizer/muon_newton_schulz_time': ns_s,
            'optimizer/muon_scatter_time': scatter_s,
            'optimizer/muon_collective_bytes': float(collective_bytes),
        }

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if self.distributed:
            self._step_distributed()
        else:
            self._step_dense()
        return loss

    def diagnostics(self):
        return {
            'optimizer/muon_parameter_count': float(self.muon_parameter_count),
            'optimizer/adamw_parameter_count': float(self.adamw_parameter_count),
            'optimizer/muon_backend_active': float(self.distributed),
            'optimizer/muon_fsdp_size': float(self.fsdp_size),
            'optimizer/muon_world_size': float(self.world_size),
            **self.last_diagnostics,
        }


def build_muon(module, optim_config, fsdp_model=None):
    """Build Muon and classify parameters using global FSDP metadata when needed."""
    distributed = fsdp_model is not None and dist.is_initialized() and dist.get_world_size() > 1
    backend = str(optim_config.get('muon_distributed_backend', 'gather_scatter')).lower()
    if backend != 'gather_scatter':
        raise ValueError(
            'unsupported Muon distributed backend; expected gather_scatter, '
            f'got {backend!r}')
    registry = build_fsdp_muon_registry(fsdp_model) if distributed else {}
    named = [(name, parameter) for name, parameter in module.named_parameters()
             if parameter.requires_grad]

    muon_params = []
    adamw_params = []
    for name, parameter in named:
        info = registry.get(parameter)
        if distributed and info is None:
            raise RuntimeError(
                f'FSDP Muon parameter {name} is missing from the validated registry; '
                'refusing an incomplete distributed update')
        shape = info.shape if info is not None else tuple(parameter.shape)
        contiguous = info.contiguous if info is not None else parameter.is_contiguous()
        is_muon = (
            len(shape) == 2 and contiguous and
            not _is_embedding_or_head(name))
        if is_muon:
            muon_params.append(parameter)
        else:
            adamw_params.append(parameter)

    if not muon_params:
        raise RuntimeError(
            'build_muon: no valid 2-D Muon parameters found; refusing an all-AdamW run')

    muon_param_ids = {id(parameter) for parameter in muon_params}
    classification = []
    for name, parameter in named:
        info = registry.get(parameter)
        canonical_name = info.name if info is not None else name
        canonical_shape = info.shape if info is not None else tuple(parameter.shape)
        classification.append((canonical_name, tuple(int(x) for x in canonical_shape),
                               'muon' if id(parameter) in muon_param_ids else 'adamw'))
    parameter_signature = hashlib.sha256(
        json.dumps(sorted(classification), separators=(',', ':')).encode('utf-8')).hexdigest()

    optimizer = Muon(
        lr=optim_config.lr,
        wd=optim_config.get('weight_decay', 1e-2),
        muon_params=muon_params,
        momentum=optim_config.get('muon_momentum', 0.95),
        nesterov=optim_config.get('muon_nesterov', True),
        ns_steps=optim_config.get('muon_ns_steps', 5),
        adamw_params=adamw_params,
        adamw_betas=tuple(optim_config.get('betas', (0.9, 0.999))),
        fsdp_model=fsdp_model,
        fsdp_registry=registry,
        parameter_signature=parameter_signature,
    )
    print(
        f'[muon] backend={optimizer.backend} '
        f'muon_params={optimizer.muon_parameter_count} '
        f'adamw_params={optimizer.adamw_parameter_count}')
    if distributed:
        print(
            f'[muon] fsdp_size={optimizer.fsdp_size} '
            f'world_size={optimizer.world_size} '
            f'sharded_parameter_registry=valid '
            f'backend_version={MUON_BACKEND_VERSION}')
    return optimizer

def demo():
    """Self-check: NS5 orthogonalizes, and build_muon splits params the way we depend on."""
    from omegaconf import OmegaConf
    from torch import nn

    torch.manual_seed(0)
    cfg = OmegaConf.create({'lr': 1e-6})

    # NS5 maps G -> ~US'V^T with S'_ii in roughly [0.5, 1.5]
    for shape in [(64, 32), (32, 64)]:
        out = zeropower_via_newtonschulz5(torch.randn(*shape), steps=5).float()
        assert out.shape == shape, out.shape
        sv = torch.linalg.svdvals(out)
        assert sv.min() > 0.4 and sv.max() < 1.6, sv

    class Toy(nn.Module):

        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(11, 8)
            self.fc1 = nn.Linear(8, 16)
            self.fc2 = nn.Linear(16, 8)
            self.norm = nn.LayerNorm(8)

    m = Toy()
    opt = build_muon(m, cfg)

    muon_flags = {n: opt.state[p]['use_muon'] for n, p in m.named_parameters()}
    assert muon_flags == {
        'embed_tokens.weight': False,  # excluded by name even though 2-D
        'fc1.weight': True,
        'fc1.bias': False,
        'fc2.weight': True,
        'fc2.bias': False,
        'norm.weight': False,
        'norm.bias': False,
    }, muon_flags
    # the regression that actually matters: a flattened model must not silently be all-AdamW
    assert sum(muon_flags.values()) == 2

    before = m.fc1.weight.detach().clone()
    loss = m.fc2(m.fc1(m.embed_tokens(torch.tensor([1, 2, 3])))).sum()
    loss.backward()
    opt.step()
    assert torch.isfinite(m.fc1.weight).all()
    assert not torch.equal(before, m.fc1.weight), 'muon step did not move the weight'

    # a flattened (1-D) parameter set must fail loudly, not fall through to AdamW
    flat = nn.Module()
    flat.flat_param = nn.Parameter(torch.randn(100))
    try:
        build_muon(flat, cfg)
    except (AssertionError, RuntimeError):
        pass
    else:
        raise AssertionError('build_muon accepted an all-1-D module')

    print('ok')


if __name__ == '__main__':
    demo()
