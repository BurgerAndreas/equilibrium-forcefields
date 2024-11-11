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

from torch_geometric.nn import radius_graph

from .gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2,
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2,
)
from .input_block import EdgeDegreeEmbedding

import deq2ff.logging_utils_deq as logging_utils_deq

# # Statistics of IS2RE 100K
# # IS2RE: 100k, max_radius = 5, max_neighbors = 100
# _AVG_NUM_NODES = 77.81317
# _AVG_DEGREE = 23.395238876342773

from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20


@registry.register_model("equiformer_v2_md17")
class EquiformerV2_MD17(EquiformerV2_OC20):
    def __init__(self, **kwargs):
        super(EquiformerV2_MD17, self).__init__(**kwargs)

    # @conditional_grad(torch.enable_grad())
    # def forward(self, data, step=None, datasplit=None, **kwargs):
    #     """The same as EquiformerV2_OC20.
    #     Encoding is simplified but yields the same results.
    #     """
    #     self.batch_size = len(data.natoms)
    #     self.dtype = data.pos.dtype
    #     self.device = data.pos.device
    #     pos = data.pos
    #     batch = data.batch

    #     data.atomic_numbers = data.z
    #     atomic_numbers = data.z.long()
    #     node_atom = atomic_numbers
    #     num_atoms = len(atomic_numbers)

    #     # encode edges
    #     # get graph edges based on radius
    #     edge_index = radius_graph(
    #         x=pos, r=self.max_radius, batch=batch, max_num_neighbors=1000
    #     )
    #     edge_src, edge_dst = edge_index[0], edge_index[1]
    #     edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)

    #     edge_distance_vec = edge_vec
    #     edge_distance = edge_distance_vec.norm(dim=-1)

    #     # E1 MD17
    #     # # radial basis function embedding of edge length
    #     # edge_length = edge_vec.norm(dim=1)
    #     # edge_length_embedding = self.rbf(edge_length)
    #     # # spherical harmonics embedding of edge vector
    #     # edge_sh = o3.spherical_harmonics(
    #     #     l=self.irreps_edge_attr,
    #     #     x=edge_vec,
    #     #     normalize=True,
    #     #     normalization="component",
    #     # )
    #     # # encode atom type z_i
    #     # atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
    #     # # Constant One, r_ij -> Linear, Depthwise TP, Linear, Scaled Scatter
    #     # edge_degree_embedding = self.edge_deg_embed(
    #     #     # atom_embedding is just used for the shape
    #     #     atom_embedding,
    #     #     edge_sh,
    #     #     edge_length_embedding,
    #     #     edge_src,
    #     #     edge_dst,
    #     #     batch,
    #     # )

    #     ###############################################################
    #     # Initialize data structures
    #     ###############################################################

    #     # Compute 3x3 rotation matrix per edge
    #     edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)

    #     # Initialize the WignerD matrices and other values for spherical harmonic calculations
    #     for i in range(self.num_resolutions):
    #         self.SO3_rotation[i].set_wigner(edge_rot_mat)

    #     ###############################################################
    #     # Initialize node embeddings
    #     ###############################################################

    #     # Init per node representations using an atomic number based embedding
    #     offset = 0
    #     x = SO3_Embedding(
    #         num_atoms,
    #         self.lmax_list,
    #         self.sphere_channels,
    #         self.device,
    #         self.dtype,
    #     )

    #     offset_res = 0
    #     offset = 0
    #     # Initialize the l = 0, m = 0 coefficients for each resolution
    #     for i in range(self.num_resolutions):
    #         if self.num_resolutions == 1:
    #             x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
    #         else:
    #             x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
    #                 :, offset : offset + self.sphere_channels
    #             ]
    #         offset = offset + self.sphere_channels
    #         offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

    #     # Edge encoding (distance and atom edge)
    #     # E1: edge_length_embedding = self.rbf(edge_length)
    #     edge_distance = self.distance_expansion(edge_distance)
    #     if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
    #         source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
    #         target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
    #         source_embedding = self.source_embedding(source_element)
    #         target_embedding = self.target_embedding(target_element)
    #         edge_distance = torch.cat(
    #             (edge_distance, source_embedding, target_embedding), dim=1
    #         )

    #     # Edge-degree embedding
    #     edge_degree = self.edge_degree_embedding(
    #         atomic_numbers, edge_distance, edge_index
    #     )
    #     # node_features = atom_embedding + edge_degree_embedding
    #     x.embedding = x.embedding + edge_degree.embedding

    #     # logging
    #     emb = x.embedding.clone().detach()

    #     ###############################################################
    #     # Update spherical node embeddings
    #     ###############################################################

    #     if self.skip_blocks:
    #         pass
    #     else:
    #         for i in range(self.num_layers):
    #             x = self.blocks[i](
    #                 x=x,  # SO3_Embedding
    #                 atomic_numbers=atomic_numbers,
    #                 edge_distance=edge_distance,
    #                 edge_index=edge_index,
    #                 batch=data.batch,  # for GraphPathDrop
    #             )

    #     # Final layer norm
    #     x.embedding = self.norm(x.embedding)

    #     ######################################################
    #     # Logging
    #     ######################################################
    #     if step is not None:
    #         logging_utils_deq.log_fixed_point_norm(
    #             x.embedding.clone().detach(), step, datasplit
    #         )
    #         # log the input injection (output of encoder)
    #         logging_utils_deq.log_fixed_point_norm(emb, step, datasplit, name="emb")

    #     ###############################################################
    #     # Energy estimation
    #     ###############################################################
    #     # (B, 16, 1)
    #     node_energy = self.energy_block(
    #         input_embedding=x,
    #         x=x,
    #         atomic_numbers=atomic_numbers,
    #         edge_distance=edge_distance,
    #         edge_index=edge_index,
    #     )
    #     # (B, 1, 1)
    #     node_energy = node_energy.embedding.narrow(dim=1, start=0, length=1)
    #     energy = torch.zeros(
    #         len(data.natoms), device=node_energy.device, dtype=node_energy.dtype
    #     )
    #     energy.index_add_(0, data.batch, node_energy.view(-1))
    #     energy = energy / self._AVG_NUM_NODES

    #     ###############################################################
    #     # Force estimation
    #     ###############################################################
    #     if self.regress_forces:
    #         forces = self.force_block(x, atomic_numbers, edge_distance, edge_index)
    #         forces = forces.embedding.narrow(1, 1, 3)
    #         forces = forces.view(-1, 3)

    #     if not self.regress_forces:
    #         return energy, {}
    #     else:
    #         return energy, forces, {}
