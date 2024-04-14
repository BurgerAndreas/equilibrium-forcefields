import torch
import omegaconf
import wandb
import copy

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm, register_norm, register_norm_module
from torchdeq.loss import fp_correction


def _init_deq(
    self,
    #implicit_layer,
    deq_mode=True,
    torchdeq_norm=omegaconf.OmegaConf.create({"norm_type": "weight_norm"}),
    deq_kwargs={},
    **kwargs,
):
    """Initializes TorchDEQ solver and normalization."""

    self.deq_mode = deq_mode
    # self.deq = get_deq(f_solver='broyden', f_max_iter=20, f_tol=1e-6)
    print(f'Passed deq_kwargs: {deq_kwargs}')
    self.deq = get_deq(**deq_kwargs)
    # self.register_buffer('z_aux', self._init_z())

    # to have weight/spectral normalization. (for better stability)
    # Using norm_type='none' in `kwargs` can also skip it.
    if torchdeq_norm.norm_type not in [None, "none", False]:
        pass
    elif "both" in torchdeq_norm.norm_type:
        norm_kwargs = copy.deepcopy(torchdeq_norm)
        norm_kwargs.pop("norm_type")
        apply_norm(self.blocks, norm_type = "weight_norm", **norm_kwargs)
        apply_norm(self.blocks, norm_type = "spectral_norm" **norm_kwargs)
        print(f'Applying both weight and spectral normalization with kwargs: {norm_kwargs}')
    else:
        apply_norm(self.blocks, **torchdeq_norm)
        # register_norm_module(DEQDotProductAttentionTransformerMD17, 'spectral_norm', names=['blocks'], dims=[0])
        print(f'Applying {torchdeq_norm.norm_type} normalization with kwargs: {torchdeq_norm}')

    return kwargs