import argparse
import datetime
import itertools
import pickle
import subprocess
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os

# from logger import FileLogger
from pathlib import Path
from typing import Iterable, Optional

import equiformer.nets as nets
from equiformer.nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from equiformer.optim_factory import create_optimizer

from equiformer.engine import AverageMeter, compute_stats

import torch
from torch_cluster import radius_graph
from torch_scatter import scatter

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm, register_norm, register_norm_module
from torchdeq.loss import fp_correction

import skimage
import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb
import omegaconf

from e3nn import o3

# import e3nn
# from e3nn.util.jit import compile_mode
# from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

from equiformer.nets.registry import register_model
from equiformer.nets.instance_norm import EquivariantInstanceNorm
from equiformer.nets.graph_norm import EquivariantGraphNorm
from equiformer.nets.layer_norm import EquivariantLayerNormV2
from equiformer.nets.radial_func import RadialProfile
from equiformer.nets.tensor_product_rescale import (
    TensorProductRescale,
    LinearRS,
    FullyConnectedTensorProductRescale,
    irreps2gate,
)
from equiformer.nets.fast_activation import Activation, Gate
from equiformer.nets.drop import (
    EquivariantDropout,
    EquivariantScalarsDropout,
    GraphDropPath,
)

from equiformer.nets.gaussian_rbf import GaussianRadialBasisLayer

# for bessel radial basis
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis

from equiformer.nets.expnorm_rbf import ExpNormalSmearing
from equiformer.nets.dp_attention_transformer import (
    ScaleFactor,
    DotProductAttention,
    DPTransBlock,
)
from equiformer.nets.graph_attention_transformer import (
    get_norm_layer,
    FullyConnectedTensorProductRescaleNorm,
    FullyConnectedTensorProductRescaleNormSwishGate,
    FullyConnectedTensorProductRescaleSwishGate,
    DepthwiseTensorProduct,
    SeparableFCTP,
    Vec2AttnHeads,
    AttnHeads2Vec,
    FeedForwardNetwork,
    NodeEmbeddingNetwork,
    ScaledScatter,
    EdgeDegreeEmbeddingNetwork,
)

from equiformer.nets.dp_attention_transformer_md17 import (
    _RESCALE,
    _MAX_ATOM_TYPE,
    _AVG_DEGREE,
    _AVG_NUM_NODES,
)

import deq2ff.logging_utils_deq as logging_utils_deq

"""
After DEQ we have to project the irreps_embeddings (all l's) to the output irreps_features (only scalars) in final_layer.
A projection head is a small alternative to a full transformer block.
"""


class IdentityBlock(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, node_input, **kwargs):
        return node_input


class FCTPProjection(torch.nn.Module):
    """See ffn_shortcut in DPTransBlock"""

    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_output,
        rescale=True,
        tp_path_norm="none",
        tp_irrep_norm=None,  # None = 'element'
        bias=True,
        # proj_drop=0.1, # 0.1
        # activation="SiLU",
        **kwargs
    ):
        super().__init__()
        self.rescale = rescale
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.proj = FullyConnectedTensorProductRescale(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_node_output,
            rescale=rescale,
            # added
            path_normalization=tp_path_norm,
            normalization=tp_irrep_norm,  # prior default: None = 'element'
            # activation=activation,
            bias=bias,
        )
        print(self.__class__.__name__, "discarded kwargs:", kwargs)

    def forward(self, node_input, node_attr, **kwargs):
        """node_input = node_features"""
        return self.proj(node_input, node_attr)


class FCTPProjectionNorm(FCTPProjection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_pre = get_norm_layer("layer")(self.irreps_node_input)
        print(self.__class__.__name__, "discarded kwargs:", kwargs)

    def forward(self, node_input, node_attr, **kwargs):
        node_input = self.norm_pre(node_input)
        return super().forward(node_input, node_attr, **kwargs)


class FFResidualFCTPProjection(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_output,
        rescale=True,
        irreps_mlp_mid=None,
        norm_layer="layer",
        # added
        tp_path_norm="none",
        tp_irrep_norm=None,  # None = 'element'
        proj_drop=0.1,  # 0.1
        bias=True,
        activation="SiLU",
        **kwargs
    ):
        super().__init__()
        self.rescale = rescale
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid)
            if irreps_mlp_mid is not None
            else self.irreps_node_input
        )
        # layer
        self.norm_layer = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            # added
            proj_drop=proj_drop,
            bias=bias,
            activation=activation,
            normalization=tp_irrep_norm,
            path_normalization=tp_path_norm,
        )
        self.ffn_shortcut = FullyConnectedTensorProductRescale(
            self.irreps_node_input,
            self.irreps_node_attr,
            self.irreps_node_output,
            # bias=True,
            rescale=rescale,
            # added
            path_normalization=tp_path_norm,
            normalization=tp_irrep_norm,  # prior default: None = 'element'
            # activation=activation,
            bias=bias,
        )
        print(self.__class__.__name__, "discarded kwargs:", kwargs)

    def forward(self, node_input, node_attr, batch, **kwargs):
        node_output = node_input
        node_features = node_input
        node_features = self.norm_layer(node_features, batch=batch)
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)
        # optionally add drop_path
        return node_output + node_features


class FFProjection(torch.nn.Module):
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_node_output,
        irreps_mlp_mid=None,
        rescale=True,
        # added
        proj_drop=0.1,  # 0.1
        bias=True,
        activation="SiLU",
        tp_irrep_norm=None,
        tp_path_norm="none",
        **kwargs
    ):
        super().__init__()
        self.rescale = rescale
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_mlp_mid = (
            o3.Irreps(irreps_mlp_mid)
            if irreps_mlp_mid is not None
            else self.irreps_node_input
        )

        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            # added
            proj_drop=proj_drop,
            bias=bias,
            activation=activation,
            normalization=tp_irrep_norm,
            path_normalization=tp_path_norm,
        )
        print(self.__class__.__name__, "discarded kwargs:", kwargs)

    def forward(self, node_input, node_attr, **kwargs):
        """node_input = node_features"""
        return self.ffn(node_input, node_attr)


class FFProjectionNorm(FFProjection):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_pre = get_norm_layer("layer")(self.irreps_node_input)

    def forward(self, node_input, node_attr, **kwargs):
        node_input = self.norm_pre(node_input)
        return super().forward(node_input, node_attr, **kwargs)


# class LinearRescaleHead(torch.nn.Module):
#     """Output head self.head
#     Only works if inputs and outputs are scalars!
#     """

#     def __init__(self, irreps_node_input, irreps_node_attr, irreps_node_output, rescale=True):
#         super().__init__()
#         self.rescale = rescale
#         self.irreps_node_input = o3.Irreps(irreps_node_input)
#         # self.irreps_node_attr = o3.Irreps(irreps_node_attr)
#         self.irreps_node_output = o3.Irreps(irreps_node_output)
#         self.proj = torch.nn.Sequential(
#             LinearRS(self.irreps_node_input, self.irreps_node_input, rescale=rescale),
#             # one activation function per irreps l
#             Activation(
#                 self.irreps_node_input,
#                 acts=[torch.nn.SiLU()] * len(self.irreps_node_input),
#             ),
#             LinearRS(self.irreps_node_input, self.irreps_node_output, rescale=rescale),
#             # out=o3.Irreps("1x0e")
#         )

#     def forward(self, node_input, **kwargs):
#         """node_input = node_features"""
#         return self.proj(node_input)
