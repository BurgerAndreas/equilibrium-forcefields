import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
import copy

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
    GraphAttention,
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


class FF(torch.nn.Module):
    def __init__(
        self, irreps_node_input, irreps_node_attr, irreps_node_output, 
        irreps_mlp_mid=None, rescale=True, 
        # added
        proj_drop=0.0, # 0.1
        bias=True,
        activation="SiLU",
        dp_tp_irrep_norm=None,
        dp_tp_path_norm="none",
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
            normalization=dp_tp_irrep_norm,
            path_normalization=dp_tp_path_norm,
        )

        print(f'FF: ignoring kwargs: {kwargs}')

    def forward(self, node_input, node_attr, **kwargs):
        """node_input = node_features"""
        return self.ffn(node_input, node_attr)

class FFNorm(FF):
    def __init__(self, affine_ln=True, **kwargs):
        super().__init__(**kwargs)
        self.norm_pre = get_norm_layer("layer")(self.irreps_node_input, affine=affine_ln)

    def forward(self, node_input, node_attr, **kwargs):
        node_input = self.norm_pre(node_input)
        return self.ffn(node_input, node_attr)

class FFResidual(torch.nn.Module):
    """Second part of DPTransBlock without norm."""
    def __init__(
        self, irreps_node_input, irreps_node_attr, irreps_node_output, irreps_mlp_mid=None, rescale=True,
        # added
        # FullyConnectedTensorProductRescale
        # only used when irreps_node_input != irreps_node_output
        fc_tp_path_norm="none",
        fc_tp_irrep_norm=None,  # None = 'element'
        proj_drop=0.1, # 0.1
        bias=True,
        activation="SiLU",
        dp_tp_irrep_norm=None,
        dp_tp_path_norm="none",
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

        self.norm_2 = None

        self.ffn = FeedForwardNetwork(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_node_output=self.irreps_node_output,
            irreps_mlp_mid=self.irreps_mlp_mid,
            # added
            proj_drop=proj_drop,
            bias=bias,
            activation=activation,
            normalization=dp_tp_irrep_norm,
            path_normalization=dp_tp_path_norm,
        )

        self.ffn_shortcut = None
        if self.irreps_node_input != self.irreps_node_output:
            # ~30k params
            self.ffn_shortcut = FullyConnectedTensorProductRescale(
                self.irreps_node_input,
                self.irreps_node_attr,
                self.irreps_node_output,
                # bias=True,
                rescale=_RESCALE,
                # added
                path_normalization=fc_tp_path_norm,
                normalization=fc_tp_irrep_norm,  # prior default: None = 'element'
                # activation=activation,
                bias=bias,
            )
        print(f'FFResidual: ignoring kwargs: {kwargs}')

    def forward(self, node_input, node_attr, batch, **kwargs):
        """node_input = node_features"""
        node_output = node_input
        node_features = node_input
        if self.norm_2 is not None:
            node_features = self.norm_2(node_features, batch=batch)  # batch unused

        # optionally reduce irreps dim
        node_features = self.ffn(node_features, node_attr)
        if self.ffn_shortcut is not None:
            node_output = self.ffn_shortcut(node_output, node_attr)

        return node_output + node_features

class FFNormResidual(FFResidual):
    """Second part of DPTransBlock."""
    def __init__(self, affine_ln=True, **kwargs):
        super().__init__(**kwargs)
        self.norm_2 = get_norm_layer("layer")(self.irreps_node_input, affine=affine_ln)

class DPA(torch.nn.Module):
    """First part of DPTransBlock without norm (if irreps_node_input=irreps_node_output)."""
    def __init__(
        self,
        irreps_node_input,
        irreps_node_attr,
        irreps_edge_attr,
        irreps_node_output,
        fc_neurons,
        irreps_head,
        num_heads,
        irreps_pre_attn=None,
        rescale_degree=False,
        # nonlinear_message=False,
        alpha_drop=0., # 0.1
        proj_drop=0., #0.1
        # drop_path_rate=0.0,
        # irreps_mlp_mid=None,
        # norm_layer="layer",
        # added
        dp_tp_path_norm="none",
        dp_tp_irrep_norm=None,  # None = 'element'
        # FullyConnectedTensorProductRescale
        # only used when irreps_node_input != irreps_node_output
        # fc_tp_path_norm="none",
        # fc_tp_irrep_norm=None,  # None = 'element'
        activation="SiLU",
        bias=True,
        **kwargs
    ):
        super().__init__()
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_pre_attn = (
            self.irreps_node_input
            if irreps_pre_attn is None
            else o3.Irreps(irreps_pre_attn)
        )
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.rescale_degree = rescale_degree
        # self.nonlinear_message = nonlinear_message
        # self.irreps_mlp_mid = (
        #     o3.Irreps(irreps_mlp_mid)
        #     if irreps_mlp_mid is not None
        #     else self.irreps_node_input
        # )

        # self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)
        self.norm_1 = None

        self.dpa = DotProductAttention(
            irreps_node_input=self.irreps_node_input,
            irreps_node_attr=self.irreps_node_attr,
            irreps_edge_attr=self.irreps_edge_attr,
            # irreps_node_output=self.irreps_node_input,
            irreps_node_output=self.irreps_node_output, # changed
            fc_neurons=fc_neurons,
            irreps_head=self.irreps_head,
            num_heads=self.num_heads,
            irreps_pre_attn=self.irreps_pre_attn,
            rescale_degree=self.rescale_degree,
            alpha_drop=alpha_drop,
            proj_drop=proj_drop,
            # added
            dp_tp_path_norm=dp_tp_path_norm,
            dp_tp_irrep_norm=dp_tp_irrep_norm,
            activation=activation,
            bias=bias,
        )

        print(f'DPA: ignoring kwargs: {kwargs}')
    
    def forward(
        self,
        node_input,
        node_attr,
        edge_src,
        edge_dst,
        edge_attr,  # requires_grad
        edge_scalars,  # requires_grad
        batch,
        **kwargs,
    ):
        # residual connection can only be applied if irreps_node_input = irreps_node_output
        # node_output = node_input
        node_features = node_input
        
        if self.norm_1 is not None:
            node_features = self.norm_1(node_features, batch=batch)  # batch unused

        # norm_1_output = node_features
        node_features = self.dpa(
            node_input=node_features,
            node_attr=node_attr,  # node_attr unused
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_attr,
            edge_scalars=edge_scalars,
            batch=batch,  # batch unused
        )

        # residual
        # return node_output + node_features
        return node_features

class DPANorm(DPA):
    """First part of DPTransBlock."""
    def __init__(self, affine_ln=True, **kwargs):
        super().__init__(**kwargs)
        self.norm_1 = get_norm_layer("layer")(self.irreps_node_input, affine=affine_ln)
        # self.norm_1 = get_norm_layer(norm_layer)(self.irreps_node_input)

# options = [DPA, DPANorm, FF, FFNorm, FFResidual, FFNormResidual]

'''
import deq2ff.logging_utils_deq as logging_utils_deq
from deq2ff.deq_equiformer.deq_equiformer_base import EquiformerDEQBase

from deq2ff.deq_equiformer.deq_dp_md17 import DEQDotProductAttentionTransformerMD17

class DEQMinimalDotProductAttention(DEQDotProductAttentionTransformerMD17):

    def __init__(self, deq_block, **kwargs) -> None:
        print(f'Using DEQ block: {deq_block}')
        self.deq_block = deq_block
        super().__init__(**kwargs)

        # assert len(self.blocks) == 1, f"Only one block is allowed for DEQMinimalDotProductAttention =/= {len(self.blocks)}"

    def build_blocks(self):
        """N blocks of: Layer Norm 1 -> DotProductAttention -> Layer Norm 2 -> FeedForwardNetwork
        Last block outputs scalars (l0) only.
        """
        for i in range(self.num_layers):
            if i == (self.num_layers - 1):
                # last block
                # last block outputs scalars only (l0 only, no higher l's)
                # irreps_node_embedding -> irreps_feature
                # "128x0e+64x1e+32x2e" -> "512x0e"
                # as of now we do not concat the node_features_input_injection
                # onto the node_features for the decoder
                irreps_node_input = self.irreps_node_embedding
                irreps_block_output = self.irreps_feature
            else:
                # first and middle layer: input injection
                irreps_node_input = self.irreps_node_z
                irreps_block_output = self.irreps_node_embedding

            if self.input_injection == "first_layer":
                if i == 0:
                    # first layer unchanged
                    pass
                elif i == (self.num_layers - 1):
                    # last layer unchanged
                    pass
                else:
                    # middle layers: no input injection
                    irreps_node_input = self.irreps_node_embedding
                    irreps_block_output = self.irreps_node_embedding
            
            # blk = DPTransBlock(
            blk = eval(self.deq_block)(
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
                # added
                dp_tp_path_norm=self.dp_tp_path_norm,
                dp_tp_irrep_norm=self.dp_tp_irrep_norm,
                fc_tp_path_norm=self.fc_tp_path_norm,
                fc_tp_irrep_norm=self.fc_tp_irrep_norm,
                activation=self.activation,
                bias=self.bias,
            )
            if i != (self.num_layers - 1):
                self.blocks.append(blk)
            else:
                self.final_block = blk
    
    

@register_model
def deq_minimal_dpa(
    **kwargs,
):
    return DEQMinimalDotProductAttention(**kwargs)
    
'''