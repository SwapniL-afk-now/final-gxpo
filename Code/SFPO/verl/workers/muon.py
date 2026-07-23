"""Muon optimizer (Moonlight / Kimi reference implementation).

Adapted from https://github.com/MoonshotAI/Moonlight (examples/toy_train.py), which is
itself a modified version of https://github.com/KellerJordan/Muon/blob/master/muon.py

Requires 2-D parameters. Under FSDP that means the actor must be built with
`use_orig_params=True` and `ShardingStrategy.NO_SHARD` -- sharded param views are
flattened to 1-D (torch `_flat_param.py:_use_sharded_views`), which would silently route
every parameter into the AdamW fallback branch below. See `build_muon`'s assertion and
`fsdp_workers._build_model_optimizer`.
"""
import math

import torch


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
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        wd: Weight decay, applied decoupled to both branches.
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

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)
        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim == 2, p.ndim
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]

            # generate weight updates
            for p in params:
                # sanity check
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf
                u = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])

                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                p.data.add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = group['lr']
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss


def build_muon(module, optim_config):
    """Split `module`'s trainable params into Muon (2-D weights) and AdamW (everything else).

    Mirrors Moonlight's `get_optimizer`: >=2-D and not embed_tokens/lm_head goes to Muon.
    The substring test survives FSDP's `_fsdp_wrapped_module.` name prefix.

    adamw_betas defaults to (0.9, 0.999) rather than Moonlight's (0.9, 0.95) so the 1-D
    params move like the AdamW baseline and the only variable is Muon on the 2-D weights.
    """

    def is_muon(name, p):
        return p.ndim >= 2 and 'embed_tokens' not in name and 'lm_head' not in name

    named = [(n, p) for n, p in module.named_parameters() if p.requires_grad]
    muon_params = [p for n, p in named if is_muon(n, p)]
    adamw_params = [p for n, p in named if not is_muon(n, p)]

    # Under FSDP with use_orig_params=False (or any sharded strategy) every param arrives
    # flattened to 1-D, which would put the whole model in the AdamW branch and silently
    # turn a "muon" run into an AdamW run. Fail loudly instead.
    assert muon_params, (
        'build_muon: no 2-D parameters found. The module is almost certainly FSDP-wrapped '
        'with flattened params -- Muon needs use_orig_params=True and ShardingStrategy.NO_SHARD.')

    print(f'[muon] muon params: {len(muon_params)} / adamw params: {len(adamw_params)}')

    return Muon(
        lr=optim_config.lr,
        wd=optim_config.get('weight_decay', 1e-2),
        muon_params=muon_params,
        momentum=optim_config.get('muon_momentum', 0.95),
        ns_steps=optim_config.get('muon_ns_steps', 5),
        adamw_params=adamw_params,
        adamw_betas=tuple(optim_config.get('betas', (0.9, 0.999))),
    )


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
    except AssertionError:
        pass
    else:
        raise AssertionError('build_muon accepted an all-1-D module')

    print('ok')


if __name__ == '__main__':
    demo()
