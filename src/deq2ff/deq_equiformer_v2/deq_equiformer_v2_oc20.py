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

# register model to be used with EquiformerV1 training loop (MD17)
from equiformer.nets.registry import register_model

from deq2ff.deq_base import _init_deq

# Statistics of IS2RE 100K
# from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import (
#     _AVG_DEGREE,
#     _AVG_NUM_NODES,
# )


"""
equiformer_v2/nets/equiformer_v2/equiformer_v2_oc20.py
"""

@registry.register_model("deq_equiformer_v2_oc20")
class DEQ_EquiformerV2_OC20(EquiformerV2_OC20):

    def __init__(
        self,
        sphere_channels,
        # deq
        sphere_channels_fixedpoint=None,
        z0="zero",
        cat_injection=True,
        norm_injection=None,
        path_norm="none",
        irrep_norm=None,
        **kwargs,
    ):
        # DEQ
        self.z0 = z0
        self.cat_injection = cat_injection
        self.norm_injection = norm_injection
        self.path_norm = path_norm # TODO: unused
        self.irrep_norm = irrep_norm # TODO: unused
        if sphere_channels_fixedpoint is None:
            sphere_channels_fixedpoint = sphere_channels
        self.sphere_channels_fixedpoint = sphere_channels_fixedpoint
        if self.cat_injection:
            assert self.sphere_channels_fixedpoint == sphere_channels

        super().__init__(sphere_channels=sphere_channels, **kwargs)

        # DEQ
        kwargs = self._init_deq(**kwargs)
        print(f"Ignoring kwargs in {self.__class__.__name__}:", kwargs)

    def build_blocks(self):
        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            sphere_channels_in = self.sphere_channels
            if i == 0:
                # add input_injection
                if self.cat_injection:
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
                norm_type=self.norm_type,
                alpha_drop=self.alpha_drop,
                drop_path_rate=self.drop_path_rate,
                proj_drop=self.proj_drop,
                # added
                normlayer_norm=self.normlayer_norm,
                normlayer_affine=self.normlayer_affine,
            )
            self.blocks.append(block)

    def _init_deq(self, **kwargs):
        return _init_deq(self, **kwargs)

    @conditional_grad(torch.enable_grad())
    def forward(self, data, step=None, datasplit=None, fixedpoint=None, return_fixedpoint=False, **kwargs):
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
        """
        # data.natoms: [batch_size]
        # data.pos: [batch_size*num_atoms, 3])
        # data.atomic_numbers = data.z: [batch_size*num_atoms]
        # data.cell: [batch_size, 3, 3]
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        if hasattr(data, "atomic_numbers"):
            atomic_numbers = data.atomic_numbers.long()
        else:
            # MD17
            atomic_numbers = data.z.long()
            data.atomic_numbers = data.z

        # When using MD17 instead of OC20
        # cell is not used unless (otf_graph is False) or (use_pbc is not None)
        if not hasattr(data, "cell"):
            data.cell = None

        # molecules in batch can be of different sizes
        num_atoms = len(atomic_numbers)
        pos = data.pos

        # basically the same as edge_src, edge_dst, edge_vec, edge_length in V1
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        # data unused
        edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x: SO3_Embedding = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )
        # both: [num_atoms, num_coefficients, num_channels]
        # num_coefficients = sum([(2 * l + 1) for l in self.lmax_list])
        # addition, not concatenation
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        # "Replaced" by DEQ
        ###############################################################

        # emb_SO3 = x
        emb = x.embedding

        if fixedpoint is None:
            x: torch.Tensor = self._init_z(shape=emb.shape)
            reuse = False
        else:
            reuse = True
            x = fixedpoint

        reset_norm(self.blocks)

        # Transformer blocks
        # f = lambda z: self.mfn_forward(z, u)
        f = lambda x: self.deq_implicit_layer(
            x,
            emb=emb,
            edge_index=edge_index,
            edge_distance=edge_distance,
            atomic_numbers=atomic_numbers,
            data=data,
        )

        # find fixed-point
        solver_kwargs = {"f_max_iter": self.fpreuse_f_max_iter} if reuse else {} 
        # returns the sampled fixed point trajectory (tracked gradients)
        # z_pred, info = self.deq(f, z, solver_kwargs=solver_kwargs)
        z_pred, info = self.deq(f, x, solver_kwargs=solver_kwargs)

        x = SO3_Embedding(
            length=num_atoms,
            lmax_list=self.lmax_list,
            num_channels=self.sphere_channels_fixedpoint,
            device=self.device,
            dtype=self.dtype,
            embedding=z_pred[-1],
        )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ######################################################
        # Logging
        ######################################################
        if step is not None:
            # log the final fixed-point
            logging_utils_deq.log_fixed_point_norm(z_pred, step, datasplit)
            # log the input injection (output of encoder)
            logging_utils_deq.log_fixed_point_norm(emb, step, datasplit, name="emb")

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x)
        node_energy = node_energy.embedding.narrow(1, 0, 1)
        energy = torch.zeros(
            len(data.natoms), device=node_energy.device, dtype=node_energy.dtype
        )
        energy.index_add_(0, data.batch, node_energy.view(-1))
        energy = energy / self._AVG_NUM_NODES

        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            # atom-wise forces using a block of equivariant graph attention
            # and treating the output of degree 1 as the predictions
            forces = self.force_block(x, atomic_numbers, edge_distance, edge_index)
            forces = forces.embedding.narrow(1, 1, 3)
            forces = forces.view(-1, 3)

        if self.regress_forces:
            if return_fixedpoint:
                # z_pred = sampled fixed point trajectory (tracked gradients)
                return energy, forces, z_pred[-1].detach().clone(), info
            return energy, forces, info
        else:
            if return_fixedpoint:
                # z_pred = sampled fixed point trajectory (tracked gradients)
                return energy, z_pred[-1].detach().clone(), info
            return energy, info

    def inject_input(self, z, u):
        if self.cat_injection:
            z = torch.cat([z, u], dim=1)
        else:
            # we can't use previous of z because we initialize z as 0
            # norm_before = z.norm()
            norm_before = u.norm()
            z = z + u
            if self.norm_injection == "prev":
                scale = z.norm() / norm_before
                z = z / scale
            elif self.norm_injection == "one":
                z = z / z.norm()
        return z

    def deq_implicit_layer(
        self, x: torch.Tensor, emb, edge_index, edge_distance, atomic_numbers, data
    ) -> torch.Tensor:
        """Implicit layer for DEQ that defines the fixed-point.
        Make sure to input and output only torch.tensor, not SO3_Embedding, to not break TorchDEQ.
        """
        # input injection
        if self.cat_injection:
            # x = torch.cat([x, emb], dim=-1)
            x = SO3_Embedding(
                length=x.shape[0],
                lmax_list=self.lmax_list,
                num_channels=self.sphere_channels + self.sphere_channels_fixedpoint,
                device=self.device,
                dtype=self.dtype,
                embedding=torch.cat([x, emb], dim=-1),
            )
        else:
            # we can't use previous of x because we initialize z as 0
            # norm_before = x.norm()
            norm_before = emb.norm()
            z = x + emb
            if self.norm_injection == "prev":
                scale = z.norm() / norm_before
                z = z / scale
            elif self.norm_injection == "one":
                z = z / z.norm()
            x = SO3_Embedding(
                length=x.shape[0],
                lmax_list=self.lmax_list,
                num_channels=self.sphere_channels,
                device=self.device,
                dtype=self.dtype,
                embedding=z,
            )
        # layers
        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,  # for GraphDropPath
            )
        return x.embedding

    def _init_z(self, shape):
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
        else:
            raise ValueError(f"Invalid z0: {self.z0}")

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) or isinstance(
                        module, SO3_LinearV2
                    ):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)


@register_model
def deq_equiformer_v2_oc20(**kwargs):
    return DEQ_EquiformerV2_OC20(**kwargs)
