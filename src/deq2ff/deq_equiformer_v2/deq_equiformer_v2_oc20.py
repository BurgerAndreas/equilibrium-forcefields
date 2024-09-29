import logging
import time
import math
import numpy as np
import torch
import torch.nn as nn

from pyexpat.model import XML_CQUANT_OPT

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass

from torch.nn import Linear
from equiformer_v2.nets.equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer
from equiformer_v2.nets.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from equiformer_v2.nets.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2,
)
from equiformer_v2.nets.equiformer_v2.module_list import ModuleListInfo
from equiformer_v2.nets.equiformer_v2.so2_ops import SO2_Convolution
from equiformer_v2.nets.equiformer_v2.radial_function import RadialFunction
from equiformer_v2.nets.equiformer_v2.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from equiformer_v2.nets.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2,
)
from equiformer_v2.nets.equiformer_v2.input_block import EdgeDegreeEmbedding
import deq2ff.logging_utils_deq as logging_utils_deq

from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

import omegaconf
import wandb
import copy

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm, register_norm, register_norm_module
from torchdeq.loss import fp_correction
from torchdeq.solver.broyden import broyden_solver_grad
from torchdeq.solver.anderson import anderson_solver
from torchdeq.solver.fp_iter import fixed_point_iter

# register model to be used with EquiformerV1 training loop (MD17)
from equiformer.nets.registry import register_model

from deq2ff.deq_base import _init_deq, _process_solver_kwargs
from deq2ff.logging_utils_deq import check_values, print_values


"""
equiformer_v2/nets/equiformer_v2/equiformer_v2_oc20.py
"""


@registry.register_model("deq_equiformer_v2_oc20")
class DEQ_EquiformerV2_OC20(EquiformerV2_OC20):
    def __init__(
        self,
        torchdeq_norm,
        deq_kwargs,
        deq_kwargs_eval,
        deq_kwargs_fpr,
        # not used but necessary for OC20 compatibility
        num_atoms=None,  # not used
        bond_feat_dim=None,  # not used
        num_targets=None,  # not used
        sphere_channels=None,
        # deq
        sphere_channels_fixedpoint=None,
        z0="zero",
        inp_inj="cat",
        inj_norm=None,
        path_norm="none",
        irrep_norm=None,
        num_layers=1,
        stacks=1,
        **kwargs,
    ):
        # DEQ
        self.z0 = z0
        self.inp_inj = inp_inj
        self.inj_norm = inj_norm
        if sphere_channels_fixedpoint is None:
            sphere_channels_fixedpoint = sphere_channels
        self.sphere_channels_fixedpoint = sphere_channels_fixedpoint

        if self.inp_inj == "cat": 
            # TODO should be if == "add"?
            # do we cat along the channel or l dimension?
            assert self.sphere_channels_fixedpoint == sphere_channels
        
        # self.stacks = stacks
        # self.num_layers = num_layers

        super().__init__(
            sphere_channels=sphere_channels, num_layers=num_layers,**kwargs
        )
        if self.inp_inj == "lc":
            # linear combination: two scalar learnable weights
            # x = w1 * x + w2 * u (u is the input injection)
            self.inj_w1 = nn.Parameter(torch.ones(1))
            self.inj_w2 = nn.Parameter(torch.ones(1))
        elif self.inp_inj == "nlc":
            # normed linear combination:
            # x = w1 * x + (1-w1) * u
            self.inj_w1 = nn.Parameter(torch.ones(1))
        elif self.inp_inj == "cwlc":
            # two matrix learnable weights # [D,C]
            # TODO: breaks equivaraince?
            num_coefficients = 0
            for i in range(self.num_resolutions):
                num_coefficients = num_coefficients + int((self.lmax_list[i] + 1) ** 2)
            self.inj_w1 = nn.Parameter(torch.ones(1, num_coefficients, sphere_channels))
            self.inj_w2 = nn.Parameter(torch.ones(1, num_coefficients, sphere_channels))
        elif self.inp_inj == "swlc":
            # sphere-channel-wise learnable weights # [C]
            self.inj_w1 = nn.Parameter(torch.ones(1, 1, sphere_channels))
            self.inj_w2 = nn.Parameter(torch.ones(1, 1, sphere_channels))

        if inj_norm == "ln":
            self.inj_norm_ln = get_normalization_layer(
                self.ln_type,
                lmax=max(self.lmax_list),
                num_channels=self.sphere_channels,
            )

        # DEQ
        self.torchdeq_norm = torchdeq_norm
        self.deq_kwargs = deq_kwargs
        self._init_deq(
            torchdeq_norm=torchdeq_norm, 
            deq_kwargs=deq_kwargs,
            deq_kwargs_eval=deq_kwargs_eval,
            deq_kwargs_fpr=deq_kwargs_fpr,
        )

    def build_blocks(self):
        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            sphere_channels_in = self.sphere_channels
            if (i == 0) and (self.inp_inj == "cat"):
                    sphere_channels_in = (
                        self.sphere_channels_fixedpoint + self.sphere_channels
                    )
            block = TransBlockV2(
                # sphere_channels=self.sphere_channels,
                sphere_channels=sphere_channels_in,
                attn_hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                ffn_hidden_channels=self.ffn_hidden_channels,
                output_channels=self.sphere_channels,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.block_use_atom_edge_embedding,
                use_m_share_rad=self.use_m_share_rad,
                attn_activation=self.attn_activation,
                use_s2_act_attn=self.use_s2_act_attn,
                use_attn_renorm=self.use_attn_renorm,
                ffn_activation=self.ffn_activation,
                use_gate_act=self.use_gate_act,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                ln_type=self.ln_type,
                # dropout
                alpha_drop=self.alpha_drop,
                path_drop=self.path_drop,
                proj_drop=self.proj_drop,
                # added
                use_variational_alpha_drop=self.use_variational_alpha_drop,
                use_variational_path_drop=self.use_variational_path_drop,
                # use_variational_proj_drop=self.use_variational_proj_drop,
                ln_norm=self.ln_norm,
                ln_affine=self.ln_affine,
                ln=self.ln,
                final_ln=self.final_ln,
            )
            # eval(f"self.{blocksname}.append(block)")
            self.blocks.append(block)

    def _init_deq(self, torchdeq_norm, deq_kwargs, **kwargs):
        return _init_deq(self, torchdeq_norm=torchdeq_norm, deq_kwargs=deq_kwargs, **kwargs)

    def set_current_deq(self, reuse=False):
        """We use different DEQ solvers for training, evaluation, and fixed-point reuse."""
        # if self.eval() and reuse:
        if reuse == True:
            self.deq_current = self.deq_eval_fpr
        elif not self.training:
            self.deq_current = self.deq_eval
        else:
            self.deq_current = self.deq

    @conditional_grad(torch.enable_grad())
    def forward(
        self,
        data,
        step=None,
        datasplit=None,
        fixedpoint=None,
        return_fixedpoint=False,
        solver_kwargs={},
        fpr_loss=False,
        reset_dropout=True,
        **kwargs,
    ):
        """
        Args:
            data: Data object containing the following attributes:
                - natoms: Number of atoms in each molecule in the batch
                - pos: Atom positions in the batch
                - z: Atomic numbers of the atoms in the batch
                - cell: Unit cell of the batch
            step: Current training step for logging
            datasplit: Data split for logging (train/val/test)
            fixedpoint: Previous fixed-point to use as initial estimate
            return_fixedpoint: Return the final fixed-point estimate
            solver_kwargs: Additional kwargs or overrides for the deq solver
        """
        # The following "encoder" is the same as EquiformerV2_OC20
        # and yields the input injection
        # data.natoms: [batch_size]
        # data.pos: [batch_size*num_atoms, 3])
        # data.atomic_numbers = data.z: [batch_size*num_atoms]
        # data.cell: [batch_size, 3, 3]
        x, pos, atomic_numbers, edge_distance, edge_index = self.encode(data)

        ###############################################################
        # Update spherical node embeddings
        # "Replaced" by DEQ
        ###############################################################

        # In Equiformer x are the node features,
        # where x is initialized in the "encoder" and then updated in the transformer blocks. 
        # In DEQ x is also initialized in the "encoder" but then used as the input injection.
        emb = x.embedding

        # if previous fixed-point is not reused, initialize z
        if fixedpoint is None:
            z: torch.Tensor = self._init_z(shape=emb.shape, emb=emb)
            reuse = False
        else:
            z = fixedpoint.to(emb.device)
            reuse = True
        # TODO:verbose
        print('z done', flush=True)

        # from torchdeq
        reset_norm(self.blocks)

        self.reset_dropout(z, data.batch)

        # Transformer blocks
        # f = lambda z: self.mfn_forward(z, u)
        def f(_z):
            # x is a tensor, not SO3_Embedding
            # if batchify_for_torchdeq is True, x in and out should be [B, N, D, C]
            return self.deq_implicit_layer(
                _z,
                # the following is input injection
                emb=emb,
                edge_index=edge_index,
                edge_distance=edge_distance,
                atomic_numbers=atomic_numbers,
                batch=data.batch,
            )

        # [B*N, D, C] -> [B, N, D, C] # torchdeq batchify
        if self.batchify_for_torchdeq:
            z = z.view(self.shape_unbatched)

        # find fixed-point
        # During training, returns the sampled fixed point trajectory (tracked gradients) according to ``n_states`` or ``indexing``.
        # During inference, returns a list containing the fixed point solution only.
        
        # # V0
        # z_pred, info = self.deq(func=f, z_init=z, solver_kwargs=solver_kwargs)
       
        # V1
        if reuse:
            z_pred, info = self.deq_eval_fpr(
                func=f, z_init=z, solver_kwargs=solver_kwargs,
            )
        elif not self.training:
            z_pred, info = self.deq_eval(
                func=f, z_init=z, solver_kwargs=solver_kwargs,
            )
        else:
            z_pred, info = self.deq(
                func=f, z_init=z, solver_kwargs=solver_kwargs,
            )
            
        # # V2
        # self.set_current_deq(reuse=reuse)
        # z_pred, info = self.deq_current(
        #     func=f, z_init=z, solver_kwargs=solver_kwargs
        # )

        # [B, N, D, C] -> [B*N, D, C] # torchdeq batchify
        if self.batchify_for_torchdeq:
            z_pred = [_z.view(self.shape_batched) for _z in z_pred]
        
        # print('z_pred.shape:', z_pred[-1])
        # print('z_pred.shape:', z_pred[-1].shape)

        ######################################################
        # Fixed-point reuse loss
        # if fpr_loss == True:
        #     # torchdeq batchify
        #     if self.deq.f_solver.__name__.startswith("broyden"):
        #         z_next, _, _info = broyden_solver_grad(
        #             func=f,
        #             x0=z_pred[-1].clone(),
        #             max_iter=1,
        #             tol=solver_kwargs.get("f_tol", self.deq.f_tol),
        #             stop_mode=solver_kwargs.get("f_stop_mode", self.deq.f_stop_mode),
        #             # return_final=True,
        #         )
        #     elif self.deq.f_solver.__name__.startswith("anderson"):
        #         z_next, _, _info = anderson_solver(
        #             func=f,
        #             x0=z_pred[-1].clone(),
        #             max_iter=1,
        #             tol=solver_kwargs.get("f_tol", self.deq.f_tol),
        #             stop_mode=solver_kwargs.get("f_stop_mode", self.deq.f_stop_mode),
        #         )
        #     elif self.deq.f_solver.__name__.startswith("fixed_point_iter"):
        #         z_next, _, _info = fixed_point_iter(
        #             func=f,
        #             x0=z_pred[-1].clone(),
        #             max_iter=1,
        #             tol=solver_kwargs.get("f_tol", self.deq.f_tol),
        #             stop_mode=solver_kwargs.get("f_stop_mode", self.deq.f_stop_mode),
        #         )
        #     else:
        #         raise ValueError(f"Invalid f_solver: {self.deq.f_solver.__name__} with fpr_loss")
        #     info["z_next"] = z_next

        ######################################################
        # Logging
        ######################################################
        # if step is not None:
        #     logging_utils_deq.log_fixed_point_norm(
        #         z_pred[-1].clone().detach(), step, datasplit
        #     )

        #     if step is not None and (step % 100 == 0) or datasplit in ["val", "test"]:
        #         self.measure_oversmoothing(x=z_pred[-1].detach(), batch=data.batch, step=step, split=datasplit)

        ###############################################################
        # Decode the fixed-point estimate
        ###############################################################

        # tensor -> S03_Embedding
        x = SO3_Embedding(
            length=self.num_atoms,
            lmax_list=self.lmax_list,
            num_channels=self.sphere_channels_fixedpoint,
            device=self.device,
            dtype=self.dtype,
            embedding=z_pred[-1],
        )
        assert torch.allclose(x.embedding, z_pred[-1])

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        print('before decode', flush=True)
        return self.decode(
            data=data,
            x=x,
            fp=z_pred[-1].detach(),  # last fixed-point estimate
            pos=pos,
            atomic_numbers=atomic_numbers,
            edge_distance=edge_distance,
            edge_index=edge_index,
            info=info,
            return_fixedpoint=return_fixedpoint,
        )

    # deprecated
    # def inject_input(self, z, u):
    #     if self.inp_inj == "cat":
    #         z = torch.cat([z, u], dim=1)
    #     elif self.inp_inj == "add":
    #         # we can't use previous of z because we initialize z as 0
    #         # norm_before = z.norm()
    #         norm_before = u.norm()
    #         z = z + u
    #         if self.inj_norm == "prev":
    #             scale = z.norm() / norm_before
    #             z = z / scale
    #         elif self.inj_norm == "one":
    #             z = z / z.norm()
    #     else:
    #         raise ValueError(f"Invalid inp_inj: {self.inp_inj}")
    #     return z

    @conditional_grad(torch.enable_grad())
    def deq_implicit_layer(
        self, z: torch.Tensor, emb, edge_index, edge_distance, atomic_numbers, batch,
        step=None, datasplit=None, solver_step=None, stack=0,
    ) -> torch.Tensor:
        """Implicit layer for DEQ that defines the fixed-point.
        Make sure to inputs and outputs are torch.tensor, not SO3_Embedding, to not break TorchDEQ.
        Args:
            z: torch.Tensor, [B, N, D, C]: fixed-point estimate (node features)
            emb: torch.Tensor, [B, N, D, C]: input injection (output of encoder)
        """
        """ Input injection """
        # [B, N, D, C] -> [B*N, D, C] # torchdeq batchify
        if self.batchify_for_torchdeq:
            z = z.view(self.shape_batched)
        # we can't use previous of x because we initialize z as 0
        # norm_before = z.norm()
        # norm_before = torch.linalg.norm(z, ord=2, dim=-1)
        # will be flattened to 1D and the 2-norm of the resulting vector will be computed
        norm_before = torch.linalg.norm(emb)
        # input injection
        channels = self.sphere_channels
        if self.inp_inj == "cat":
            # z = torch.cat([z, emb], dim=-1)
            z = torch.cat([z, emb], dim=-1)
            channels = self.sphere_channels + self.sphere_channels_fixedpoint
        elif self.inp_inj == "add":
            z = z + emb
        elif self.inp_inj == "lc":
            # linear combination with scalar weights
            z = self.inj_w1 * z + self.inj_w2 * emb
        elif self.inp_inj == "nlc":
            # linear combination with scalar weights
            z = self.inj_w1 * z + (1-self.inj_w1) * emb
        elif self.inp_inj == "cwlc":
            # linear combination with matrix weights
            # inj_w1: [D,C] -> [B*N, D, C]
            # inj_w1 = self.inj_w1.unsqueeze(0).expand(x.shape[0], -1, -1)
            inj_w1 = self.inj_w1.repeat(z.shape[0], 1, 1)
            inj_w2 = self.inj_w2.repeat(z.shape[0], 1, 1)
            z = inj_w1 * z + inj_w2 * emb
        elif self.inp_inj == "swlc":
            inj_w1 = self.inj_w1.repeat(z.shape[0], z.shape[1], 1)
            inj_w2 = self.inj_w2.repeat(z.shape[0], z.shape[1], 1)
            z = inj_w1 * z + inj_w2 * emb
        else:
            raise ValueError(f"Invalid inp_inj: {self.inp_inj}")
        
        """ Normalize after input injection """
        # print_values(z, "injprenorm", log=False)
        if self.inj_norm == "prev":
            scale = torch.linalg.norm(z) / norm_before
            z = z / scale
        elif self.inj_norm == "one":
            z = z / torch.linalg.norm(z)
        elif self.inj_norm == "ln":
            # z = self.inj_ln(z)
            z = self.inj_norm_ln(z)
            # raise NotImplementedError("inj_norm=ln: use enc_ln=True inj_norm=null instead.")
        elif self.inj_norm in [None, False, "none", "None"]:
            pass
        else:
            raise ValueError(f"Invalid inj_norm: {self.inj_norm}")
        z = SO3_Embedding(
            length=z.shape[0],
            lmax_list=self.lmax_list,
            num_channels=channels,
            device=self.device,
            dtype=self.dtype,
            embedding=z,
        )
        # print_values(x.embedding, "postinj", log=False)

        """ Layers / Transformer blocks """
        # prev_layers = self.num_layers_per_stack * stack
        # for i in range(prev_layers, prev_layers + self.num_layers_per_stack):
        # print('before blocks', flush=True)
        for i in range(self.num_layers):
            z = self.blocks[i](
                z,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=batch,  # for GraphPathDrop
            )
            # self.cnt_layer += 1
            # self.measure_oversmoothing(x=x.embedding, batch=data.batch, step=step, split=datasplit, layer=self.cnt_layer)
        z = z.embedding
        # [B*N, D, C] -> [B, N, D, C] # torchdeq batchify
        if self.batchify_for_torchdeq:
            z = z.view(self.shape_unbatched)
        return z

    def _init_z(self, shape, emb=None):
        """Initializes fixed-point for DEQ
        shape: [num_atoms * batch_size, irreps_dim]
        irreps_dim = a*1 + b*3 + c*5
        """
        if self.z0 == "zero":
            return torch.zeros(
                # [batch_size, dim],
                shape,
                device=self.device,
            )
        elif self.z0 == "one":
            return torch.ones(
                # [batch_size, dim],
                shape,
                device=self.device,
            )
        elif self.z0 == "emb":
            return emb
        elif self.z0.startswith('rand'):
            # rand_0.1
            if len(self.z0.split('_')) > 1:
                mult = float(self.z0.split('_')[1])
            else:
                mult = 1.0
            return torch.rand(shape, device=self.device) * mult
        elif self.z0.startswith('normal'):
            # normal_mean_std
            return torch.normal(
                mean = float(self.z0.split('_')[1]),
                std = float(self.z0.split('_')[2]),
                size = shape,
                device = self.device,
            )
        else:
            raise ValueError(f"Invalid z0: {self.z0}")


@register_model
def deq_equiformer_v2_oc20(**kwargs):
    return DEQ_EquiformerV2_OC20(**kwargs)
