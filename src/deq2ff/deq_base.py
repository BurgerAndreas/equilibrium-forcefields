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
    torchdeq_norm,  # omegaconf.OmegaConf
    deq_kwargs,  # {}
    deq_kwargs_eval,  # {}
    deq_kwargs_fpr,
    **kwargs,
):
    """Initializes TorchDEQ solver and normalization."""

    self.deq_mode = True
    # print(f"Passed deq_kwargs: {deq_kwargs}")
    # self.deq = get_deq(f_solver='broyden', f_max_iter=20, f_tol=1e-6)
    self.kwargs_deq = deq_kwargs.copy()
    self.deq = get_deq(**deq_kwargs, name="deq_train")

    kwargs_deq1 = copy.deepcopy(deq_kwargs)
    kwargs_deq1.update(deq_kwargs_eval)
    self.kwargs_deq_eval = kwargs_deq1.copy()
    self.deq_eval = get_deq(**kwargs_deq1, name="deq_eval")

    kwargs_deq2 = copy.deepcopy(deq_kwargs)
    # kwargs_deq2.update(deq_kwargs_eval)
    kwargs_deq2.update(deq_kwargs_fpr)
    self.kwargs_deq_fpr = kwargs_deq2.copy()
    self.deq_eval_fpr = get_deq(**kwargs_deq2, name="deq_eval_fpr")

    self.deq_current = self.deq

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


# deprecated
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
