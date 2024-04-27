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

def equiformerv2(
    num_atoms=None,  # not used
    bond_feat_dim=None,  # not used
    num_targets=None,  # not used
    use_pbc=True,
    regress_forces=True,
    otf_graph=True,
    max_neighbors=500,
    max_radius=5.0,
    max_num_elements=90,
    num_layers=12,
    sphere_channels=128,
    attn_hidden_channels=128,
    num_heads=8,
    attn_alpha_channels=32,
    attn_value_channels=16,
    ffn_hidden_channels=512,
    norm_type="rms_norm_sh",
    # num_coefficients = sum_i int((lmax_list[i] + 1) ** 2)
    # lmax_list=[3] -> num_coefficients = 16
    lmax_list=[6],
    mmax_list=[2],
    grid_resolution=None,
    num_sphere_samples=128,
    edge_channels=128,
    use_atom_edge_embedding=True,
    share_atom_edge_embedding=False,
    use_m_share_rad=False,
    distance_function="gaussian",
    num_distance_basis=512,
    attn_activation="scaled_silu",
    use_s2_act_attn=False,
    use_attn_renorm=True,
    ffn_activation="scaled_silu",
    use_gate_act=False,
    use_grid_mlp=False,
    use_sep_s2_act=True,
    alpha_drop=0.1,
    drop_path_rate=0.05,
    proj_drop=0.0,
    weight_init="normal",
    # added
    # Statistics of IS2RE 100K
    # IS2RE: 100k, max_radius = 5, max_neighbors = 100
    _AVG_NUM_NODES=77.81317,
    _AVG_DEGREE=23.395238876342773,
    task_mean=None,
    task_std=None,
    name=None,
    force_head="SO2EquivariantGraphAttention",
    energy_head="FeedForwardNetwork",
    # added
    normlayer_norm="norm",
    normlayer_affine=True,
    **kwargs,
):
    # self.dtype = data.pos.dtype
    # self.device = data.pos.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # set device
    torch.set_default_dtype(dtype)
    # torch.set_default_tensor_type(torch.FloatTensor)
    torch.set_default_device(device)

    # num_atoms = len(atomic_numbers)
    num_atoms = 10

    """ What does normalization layer do to a random tensor? """
    # layer_norm_sh, layer_norm, rms_norm_sh
    for norm_type in ["layer_norm_sh", "layer_norm", "rms_norm_sh"]:
        """layer_norm_sh: 
        Args:
            affine=True 
            normalization="component" # component, norm
        Params:
            .affine_weight, .balance_degree_weight
            norm_1: layer_norm = norm_l0.weight, .bias
        """
        """layer_norm:
        Args:
            affine=True
            normalization="component" # component, norm
        Params:
            .affine_weight, .affine_bias
        """
        """rms_norm_sh:
        Args:
            affine=True
            normalization="component" # component, norm
        Params:
            .affine_weight, .affine_bias, .balance_degree_weight
            norm_1: layer_norm = norm_l0.weight, .bias
        """
        print(f'\nNormalization layer: {norm_type}')

        x = SO3_Embedding(
            num_atoms,
            lmax_list,
            sphere_channels,
            device,
            dtype,
        )
        print(f"SO3_Embedding: {x.embedding.shape}")

        x.embedding = torch.rand_like(x.embedding)
        print(f'before norm layer: fro={x.embedding.norm().item()}, l1={x.embedding.norm(1).item()}, l2={x.embedding.norm(2).item()}')

        # TransBlockV2 style
        max_lmax = max(lmax_list)
        norm_1 = get_normalization_layer(
            norm_type, lmax=max_lmax, num_channels=sphere_channels,
            normalization=normlayer_norm, affine=normlayer_affine
        )

        # forward
        x.embedding = norm_1(x.embedding)
        print(f'after norm layer: fro={x.embedding.norm().item()}, l1={x.embedding.norm(1).item()}, l2={x.embedding.norm(2).item()}')


if __name__ == "__main__":
    equiformerv2()