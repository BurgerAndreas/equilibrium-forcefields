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

import equiformer.datasets.pyg.md17 as md17_dataset

import equiformer.nets as nets
from equiformer.nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from equiformer.optim_factory import create_optimizer

from equiformer.engine import AverageMeter, compute_stats

import torch
import torch.nn as nn
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

import deq2ff.deq_utils as deq_utils

    
class FCTPProjection(nn.Module):
    """See ffn_shortcut in DPTransBlock"""
    def __init__(self, irreps_in, irreps_node_attr, irreps_out, rescale=True):
        super().__init__()
        self.rescale = rescale
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_output = o3.Irreps(irreps_out)
        self.proj = FullyConnectedTensorProductRescale(
            self.irreps_node_input, self.irreps_node_attr, self.irreps_node_output, rescale=rescale
        )

    def forward(self, node_input, node_attr, **kwargs):
        """node_input = node_features"""
        return self.proj(node_input, node_attr)
    
class ResidualFCTPProjection(FCTPProjection):
    def forward(self, node_input, node_attr, **kwargs):
        return node_input + self.proj(node_input, node_attr)

class FFProjection(nn.Module):
    def __init__(self, irreps_in, irreps_node_attr, irreps_out, irreps_mlp_mid=None, rescale=True):
        super().__init__()
        self.rescale = rescale
        self.irreps_node_input = o3.Irreps(irreps_in)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_output = o3.Irreps(irreps_out)
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
            # proj_drop=proj_drop,
        )
     
    def forward(self, node_input, node_attr, **kwargs):
        """node_input = node_features"""
        return self.ffn(node_input, node_attr)

class ResidualFFProjection(FFProjection):
    def forward(self, node_features, node_attr, **kwargs):
        return node_features + self.ffn(node_features, node_attr)

class LinearRescaleHead(nn.Module):
    """Output head self.head"""
    def __init__(self, irreps_in, irreps_node_attr, irreps_out, rescale=True):
        super().__init__()
        self.rescale = rescale
        self.irreps_node_input = o3.Irreps(irreps_in)
        # self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_node_output = o3.Irreps(irreps_out)
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_node_input, self.irreps_node_input, rescale=rescale),
            Activation(self.irreps_node_input, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_node_input, self.irreps_node_output, rescale=rescale),
        )

    def forward(self, node_input, **kwargs):
        """node_input = node_features"""
        return self.head(node_input)



from .deq_dp_md17 import DEQDotProductAttentionTransformerMD17

class DEQDecProjHeadDotProductAttentionTransformerMD17(DEQDotProductAttentionTransformerMD17):
    """
    DEQDotProductAttentionTransformerMD17 but with final_block moved from the decoder to the implicit layer.
    "LinearRescaleHead", 
    "ResidualFFProjection", "FFProjection" 
    "ResidualFCTPProjection", "FCTPProjection", 
    """

    def __init__(
        self,
        dec_proj='FFProjection',
        **kwargs,
    ):
        super().__init__(**kwargs)

        # decoder_proj
        self.final_block = eval(dec_proj)(
            irreps_in = self.irreps_node_embedding,
            irreps_node_attr = self.irreps_node_attr,
            irreps_out = self.irreps_feature,
        )

    def build_blocks(self):
        """N blocks of: Layer Norm 1 -> DotProductAttention -> Layer Norm 2 -> FeedForwardNetwork
        Last block outputs scalars (l0) only.
        """
        for i in range(self.num_layers):
            irreps_node_input = self.irreps_node_z
            irreps_block_output = self.irreps_node_embedding
            # Layer Norm 1 -> DotProductAttention -> Layer Norm 2 -> FeedForwardNetwork
            # extra stuff (= everything except node_features) is used for KV in DotProductAttention
            blk = DPTransBlock(
                # irreps_node_input=self.irreps_node_embedding,
                irreps_node_input=irreps_node_input,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                # output: which l's?
                irreps_node_output=irreps_block_output,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer,
            )
            self.blocks.append(blk)



@register_model
def deq_decprojhead_dot_product_attention_transformer_exp_l2_md17(
    irreps_in,
    radius,
    num_layers=6,
    num_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    irreps_node_attr="1x0e",
    basis_type="exp",
    # most import for parameter count?
    fc_neurons=[64, 64],
    irreps_node_embedding_injection="64x0e+32x1e+16x2e",
    irreps_node_embedding="128x0e+64x1e+32x2e",
    irreps_feature="512x0e",  # scalars only
    irreps_sh="1x0e+1x1e+1x2e",
    irreps_head="32x0e+16x1e+8x2e",
    num_heads=4,
    irreps_mlp_mid="384x0e+192x1e+96x2e",
    #
    irreps_pre_attn=None,
    rescale_degree=False,
    nonlinear_message=False,
    norm_layer="layer",
    alpha_drop=0.0,
    proj_drop=0.0,
    out_drop=0.0,
    drop_path_rate=0.0,
    scale=None,
    deq_kwargs={},
    torchdeq_norm=omegaconf.OmegaConf.create({'norm_type': 'weight_norm'}),
    init_z_from_enc=True,
    **kwargs,
):
    # dot_product_attention_transformer_exp_l2_md17
    model = DEQDecProjHeadDotProductAttentionTransformerMD17(
        irreps_in=irreps_in,
        num_layers=num_layers,
        irreps_node_attr=irreps_node_attr,
        max_radius=radius,
        number_of_basis=num_basis,
        basis_type=basis_type,
        # most import for parameter count?
        fc_neurons=fc_neurons,
        irreps_node_embedding_injection=irreps_node_embedding_injection,
        irreps_node_embedding=irreps_node_embedding,
        irreps_feature=irreps_feature,
        irreps_sh=irreps_sh,
        irreps_head=irreps_head,
        num_heads=num_heads,
        irreps_mlp_mid=irreps_mlp_mid,
        #
        irreps_pre_attn=irreps_pre_attn,
        rescale_degree=rescale_degree,
        nonlinear_message=nonlinear_message,
        norm_layer=norm_layer,
        alpha_drop=alpha_drop,
        proj_drop=proj_drop,
        out_drop=out_drop,
        drop_path_rate=drop_path_rate,
        mean=task_mean,
        std=task_std,
        scale=scale,
        atomref=atomref,
        # DEQ specific
        deq_mode=True,
        deq_kwargs=deq_kwargs,
        torchdeq_norm=torchdeq_norm,
        init_z_from_enc=init_z_from_enc,
    )
    print(f"! Ignoring kwargs: {kwargs}")
    return model
