import torch
import omegaconf
import wandb
import copy

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm, register_norm, register_norm_module
from torchdeq.loss import fp_correction

from e3nn import o3

class DEQBase:
    def _set_deq_vars(
        self,
        irreps_node_embedding,
        input_injection="first_layer",  # False=V1, 'first_layer'=V2
        irreps_node_embedding_injection="64x0e+32x1e+16x2e",
        z0="zero",
        log_fp_error_traj=False,
        dp_tp_path_norm="none",
        dp_tp_irrep_norm=None, # None = 'element'
        fc_tp_path_norm="none",
        fc_tp_irrep_norm=None, # None = 'element'
        activation='SiLU',
        **kwargs,
    ):

        self.dp_tp_path_norm = dp_tp_path_norm
        self.dp_tp_irrep_norm = dp_tp_irrep_norm
        self.fc_tp_path_norm = fc_tp_path_norm
        self.fc_tp_irrep_norm = fc_tp_irrep_norm
        self.activation = activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_injection = input_injection
        if input_injection is False:
            # V1: node_features are initialized as the output of the encoder
            self.irreps_node_injection = o3.Irreps(
                irreps_node_embedding
            )  # output of encoder
            self.irreps_node_z = o3.Irreps(irreps_node_embedding)  # input to block
            self.irreps_node_embedding = o3.Irreps(
                irreps_node_embedding
            )  # output of block
        elif self.input_injection in ["first_layer", "every_layer", True, "legacy"]:
            # V2: node features are initialized as 0
            # and the node features from the encoder are used as input injection
            # encoder = atom_embed() and edge_deg_embed()
            # both encoder shapes are defined by irreps_node_embedding
            # input to self.blocks is the concat of node_input and node_injection
            self.irreps_node_injection = o3.Irreps(
                irreps_node_embedding_injection
            )  # output of encoder
            self.irreps_node_embedding = o3.Irreps(
                irreps_node_embedding
            )  # output of block
            # "128x0e+64x1e+32x2e" + "64x0e+32x1e+16x2e"
            # 128x0e+64x1e+32x2e+64x0e+32x1e+16x2e
            irreps_node_z = self.irreps_node_embedding + self.irreps_node_injection
            irreps_node_z.simplify()
            self.irreps_node_z = o3.Irreps(irreps_node_z).simplify()  # input to block
        else:
            raise ValueError(f"Invalid input_injection: {input_injection}")

        self.z0 = z0
        # tables to log to wandb
        self.log_fp_error_traj = log_fp_error_traj
        self.fp_error_traj = {"train": None, "val": None, "test": None}

        self.dec_proj = None

        return kwargs

    def _init_deq(self, deq_mode=True,
        torchdeq_norm=omegaconf.OmegaConf.create({"norm_type": "weight_norm"}),
        deq_kwargs={}, **kwargs):
        #################################################################
        # DEQ specific

        self.deq_mode = deq_mode
        self.deq = get_deq(**deq_kwargs)
        # self.deq = get_deq(f_solver='broyden', f_max_iter=20, f_tol=1e-6)
        # self.register_buffer('z_aux', self._init_z())

        # from DEQ INR example
        # https://colab.research.google.com/drive/12HiUnde7qLadeZGGtt7FITnSnbUmJr-I?usp=sharing#scrollTo=RGgPMQLT6IHc
        # https://github.com/locuslab/torchdeq/blob/main/deq-zoo/ignn/graphclassification/layers.py
        # This function automatically decorates weights in your DEQ layer
        # to have weight/spectral normalization. (for better stability)
        # Using norm_type='none' in `kwargs` can also skip it.
        if torchdeq_norm.norm_type not in [None, "none", False]:
            pass
        elif 'both' in torchdeq_norm.norm_type:
            norm_kwargs = copy.deepcopy(torchdeq_norm)
            norm_kwargs.norm_type = 'spectral_norm'
            apply_norm(self.blocks, **norm_kwargs)
            norm_kwargs.norm_type = 'weight_norm'
            apply_norm(self.blocks, **norm_kwargs)
        else:
            apply_norm(self.blocks, **torchdeq_norm)
            # register_norm_module(DEQDotProductAttentionTransformerMD17, 'spectral_norm', names=['blocks'], dims=[0])
        
        return kwargs
    