import torch
import omegaconf
import wandb
import copy

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm, register_norm, register_norm_module
from torchdeq.loss import fp_correction

from e3nn import o3

from deq2ff.deq_base import _init_deq


class EquiformerDEQBase:
    def _set_deq_vars(
        self,
        irreps_node_embedding,
        input_injection="first_layer",  # False=V1, 'first_layer'=V2
        cat_injection=False,
        norm_injection=None,
        irreps_node_embedding_injection="64x0e+32x1e+16x2e",
        limit_f_max_iter_fpreuse=False,
        z0="zero",
        z0_requires_grad=False,
        log_fp_error_traj=False,
        dp_tp_path_norm="none",
        dp_tp_irrep_norm=None,  # None = 'element'
        fc_tp_path_norm="none",
        fc_tp_irrep_norm=None,  # None = 'element'
        activation="SiLU",
        # blocks
        dec_proj=None,
        deq_block=None,
        # weight initialization
        weight_init=None,
        weight_init_blocks=None,
        bias=True,
        affine_ln=True,
        # debugging
        skip_implicit_layer=False,
        **kwargs,
    ):
        """Sets extra variables we have added for the DEQ model.

        Args:
            weight_init (str | dict): weight initialization method. Will default to 'equiformer' for all unspecified keys.
                Keys: 'EquivariantLayerNormV2_w', 'EquivariantLayerNormV2_b', 'LayerNorm_w', 'LayerNorm_b', 'Linear_w', 'Linear_b', 'ParameterList'.
                Values: 'equiformer', 'torch', <float>, normal_<mean>_<std>, uniform_<low>_<high>.
                python scripts/deq_equiformer.py model.weight_init_blocks='{EquivariantLayerNormV2_w:1,EquivariantLayerNormV2_b:normal_0.0_0.1}'
        """

        self.dp_tp_path_norm = dp_tp_path_norm
        self.dp_tp_irrep_norm = dp_tp_irrep_norm
        self.fc_tp_path_norm = fc_tp_path_norm
        self.fc_tp_irrep_norm = fc_tp_irrep_norm
        self.activation = activation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dec_proj = dec_proj
        self.deq_block = deq_block
        self.limit_f_max_iter_fpreuse = limit_f_max_iter_fpreuse
        self.affine_ln = affine_ln

        # concat input injection or add it to the node features (embeddings)
        self.cat_injection = cat_injection
        if cat_injection is False:
            if irreps_node_embedding_injection != irreps_node_embedding:
                irreps_node_embedding_injection = irreps_node_embedding
                print(
                    f"Warning: `cat_injection` is False and addition is used. " 
                    f"Setting `irreps_node_embedding_injection` = `irreps_node_embedding` = " 
                    f"{irreps_node_embedding}."
                )
                # assert input_injection in ["first_layer", "every_layer", True, "legacy"]

        self.input_injection = input_injection
        if input_injection is False:
            # V1: node_features are initialized as the output of the encoder
            # output of encoder
            self.irreps_node_injection = o3.Irreps(irreps_node_embedding)  
            # input to block
            self.irreps_node_z = o3.Irreps(irreps_node_embedding)  
            # output of block
            self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)  
        elif self.input_injection in ["first_layer", "every_layer", True, "legacy"]:
            # V2: node features are initialized as 0
            # and the node features from the encoder are used as input injection
            # encoder = atom_embed() and edge_deg_embed()
            # both encoder shapes are defined by irreps_node_embedding
            # input to self.blocks is the concat of node_input and node_injection
            # output of encoder
            self.irreps_node_injection = o3.Irreps(irreps_node_embedding_injection)  
            # output of block
            self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)  
            # "128x0e+64x1e+32x2e" + "64x0e+32x1e+16x2e"
            # 128x0e+64x1e+32x2e+64x0e+32x1e+16x2e
            # input to block
            if self.cat_injection:
                irreps_node_z = self.irreps_node_embedding + self.irreps_node_injection
                self.irreps_node_z = o3.Irreps(irreps_node_z) 
            else:
                self.irreps_node_z = self.irreps_node_embedding
        else:
            raise ValueError(f"Invalid input_injection: {input_injection}")

        self.z0 = z0
        self.z0_requires_grad = z0_requires_grad
        # deprecated: tables to log to wandb (should be False and removed in the future)
        self.log_fp_error_traj = log_fp_error_traj
        self.fp_error_traj = {
            "train": None,
            "val": None,
            "test": None,
            "test_final": None,
        }

        self.bias = bias

        # weight initialization
        weight_init_keys = [
            "EquivariantLayerNormV2_w",
            "EquivariantLayerNormV2_b",
            "LayerNorm_w",
            "LayerNorm_b",
            "Linear_w",
            "Linear_b",
            "ParameterList",
        ]
        if isinstance(weight_init, omegaconf.dictconfig.DictConfig):
            weight_init = dict(weight_init)
            # weight_init = omegaconf.OmegaConf.to_container(weight_init)
        if isinstance(weight_init_blocks, omegaconf.dictconfig.DictConfig):
            weight_init_blocks = dict(weight_init_blocks)
            # weight_init_blocks = omegaconf.OmegaConf.to_container(weight_init_blocks)
        #
        if weight_init is None:
            self.weight_init = {k: "equiformer" for k in weight_init_keys}
        elif isinstance(weight_init, str):
            self.weight_init = {k: weight_init for k in weight_init_keys}
        elif isinstance(weight_init_blocks, dict):
            self.weight_init = {k: "equiformer" for k in weight_init_keys}
            self.weight_init.update(weight_init)
        else:
            raise ValueError(f"Invalid weight_init: {weight_init} ({type(weight_init)}")
        # weight_init_blocks will overwrite weight_init for the blocks
        if weight_init_blocks is None:
            self.weight_init_blocks = {k: "equiformer" for k in weight_init_keys}
        elif isinstance(weight_init, str):
            self.weight_init_blocks = {k: weight_init for k in weight_init_keys}
        elif isinstance(weight_init_blocks, dict):
            self.weight_init_blocks = {k: "equiformer" for k in weight_init_keys}
            self.weight_init_blocks.update(weight_init_blocks)
        else:
            raise ValueError(
                f"Invalid weight_init_blocks: {weight_init_blocks} ({type(weight_init_blocks)}"
            )
        # update wandb config
        if wandb.run is not None:
            wandb.config.update({"model.weight_init": self.weight_init})
            wandb.config.update(
                {"model.weight_init_blocks": self.weight_init_blocks}
            )

        self.skip_implicit_layer = skip_implicit_layer

        return kwargs

    def _init_decoder_proj_final_layer(self):
        """
        After DEQ we have to project the irreps_embeddings (all l's) to the output irreps_features (only scalars) in final_layer.
        A projection head is a small alternative to a full transformer block.
        """
        if self.dec_proj is not None:
            from deq2ff.deq_equiformer.deq_decprojhead_dp_md17 import (
                FFProjection,
                FFProjectionNorm,
                FFResidualFCTPProjection,
                FCTPProjection,
                FCTPProjectionNorm,
            )

            # decoder projection head
            assert (
                self.input_injection == "first_layer"
            ), "Only `input_injection='first_layer'` is supported."

            # add a layer norm?
            # self.norm_after_deq = get_norm_layer(self.norm_layer)(self.irreps_feature)

            self.final_block = eval(self.dec_proj)(
                irreps_in=self.irreps_node_embedding,
                irreps_node_attr=self.irreps_node_attr,
                irreps_out=self.irreps_feature,
            )
            self.final_block.apply(self._init_weights)
            print(
                f"\nInitialized decoder projection head `{self.dec_proj}` with {sum(p.numel() for p in self.final_block.parameters() if p.requires_grad)} parameters."
            )

    def _init_deq(self, **kwargs):
        """Initializes TorchDEQ."""
        return _init_deq(self, **kwargs)
