import torch
import omegaconf
import wandb
import copy

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm, register_norm, register_norm_module
from torchdeq.loss import fp_correction


def _init_deq(
    self,
    # implicit_layer,
    torchdeq_norm, # omegaconf.OmegaConf
    deq_kwargs, # {}
    **kwargs,
):
    """Initializes TorchDEQ solver and normalization."""

    self.deq_mode = True
    # print(f"Passed deq_kwargs: {deq_kwargs}")
    # self.deq = get_deq(f_solver='broyden', f_max_iter=20, f_tol=1e-6)
    self.deq = get_deq(**deq_kwargs)

    # weight/spectral normalization for better stability
    # Using norm_type='none' in `kwargs` can also skip it.
    if torchdeq_norm["norm_type"] in [None, "none", False]:
        print(f"Not applying torchdeq normalization.")
        pass

    else:
        print(
            f"Applying {torchdeq_norm['norm_type']} normalization with kwargs: {torchdeq_norm}"
        )
        apply_norm(self.blocks, **torchdeq_norm)
        # register_norm_module(DEQDotProductAttentionTransformerMD17, 'spectral_norm', names=['blocks'], dims=[0])

    return kwargs


def _process_solver_kwargs(solver_kwargs, reuse=False):
    _solver_kwargs = {}
    # kwargs during inference
    for k, v in solver_kwargs.items():
        # kwargs that are only used when reusing the fixed-point
        if k.startswith("fpreuse_"):
            if reuse == False:
                continue
            k = k.replace("fpreuse_", "")
        # add kwargs to solver
        if v != "_default":
            _solver_kwargs[k] = v
    return _solver_kwargs
