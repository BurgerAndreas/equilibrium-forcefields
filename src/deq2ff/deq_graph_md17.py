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
from torchdeq.norm import apply_norm, reset_norm
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

# for bessel radial basis
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis

from equiformer.nets.registry import register_model
from equiformer.nets.graph_attention_transformer_md17 import _RESCALE, _USE_BIAS, _MAX_ATOM_TYPE, _AVG_DEGREE, _AVG_NUM_NODES, CosineCutoff, ExpNormalSmearing
from equiformer.nets.fast_activation import Activation, Gate
from equiformer.nets.tensor_product_rescale import LinearRS
from equiformer.nets.instance_norm import EquivariantInstanceNorm
from equiformer.nets.graph_norm import EquivariantGraphNorm
from equiformer.nets.layer_norm import EquivariantLayerNormV2
from equiformer.nets.drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath
from equiformer.nets.gaussian_rbf import GaussianRadialBasisLayer
from equiformer.nets.graph_attention_transformer import (
    get_norm_layer,
    FullyConnectedTensorProductRescaleNorm,
    FullyConnectedTensorProductRescaleNormSwishGate,
    FullyConnectedTensorProductRescaleSwishGate,
    DepthwiseTensorProduct,
    SeparableFCTP,
    Vec2AttnHeads,
    AttnHeads2Vec,
    GraphAttention,
    FeedForwardNetwork,
    TransBlock,
    NodeEmbeddingNetwork,
    EdgeDegreeEmbeddingNetwork,
    ScaledScatter,
)

import deq2ff.deq_utils as deq_utils

from deq2ff.deq_dp_md17 import DEQDotProductAttentionTransformerMD17

# class DEQGraphAttentionTransformerMD17(torch.nn.Module):
class DEQGraphAttentionTransformerMD17(DEQDotProductAttentionTransformerMD17):
    """
    Modified from equiformer.nets.graph_attention_transformer_md17.GraphAttentionTransformerMD17

    Gets from DEQDotProductAttentionTransformerMD17:
    forward, encode, deq_implicit_layer,
    no_weight_decay, _init_weights, init_z

    Only difference to between GraphAttention and DotProductAttention:
    - use_attn_head: output head can be GraphAttention or MLP
    - blocks contain TransBlock instead of DPTransBlock
    """

    def __init__(
        self,
        use_attn_head=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.use_attn_head = use_attn_head

        # Output head
        if self.use_attn_head:
            self.head = GraphAttention(
                irreps_node_input=self.irreps_feature,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                irreps_node_output=o3.Irreps("1x0e"),
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
            )
            self.apply(self._init_weights)
        else:
            pass
            # already initialized in DEQDotProductAttentionTransformerMD17
            # self.head = torch.nn.Sequential(
            #     LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
            #     Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            #     LinearRS(self.irreps_feature, o3.Irreps("1x0e"), rescale=_RESCALE),
            # )


    def build_blocks(self):
        for i in range(self.num_layers):
            if i != (self.num_layers - 1):
                irreps_node_input = self.irreps_node_z
                irreps_block_output = self.irreps_node_embedding
            else:
                irreps_node_input = self.irreps_node_embedding
                irreps_block_output = self.irreps_feature
            blk = TransBlock(
                irreps_node_input=irreps_node_input,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
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
            if i != (self.num_layers - 1):
                self.blocks.append(blk)
            else:
                self.final_block = blk

    @torch.enable_grad()
    def decode(self, node_features, u, batch, pos):
        """Decode the node features into energy and forces (scalars).
        Basically the last third of GraphAttentionTransformerMD17.forward()
        """

        node_attr, edge_src, edge_dst, edge_sh, edge_length_embedding, _ = u
        node_features = self.final_block(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_scalars=edge_length_embedding,
            batch=batch,
        )

        # [num_atoms*batch_size, irreps_dim] -> [num_atoms*batch_size, irreps_dim]
        node_features = self.norm(node_features, batch=batch)
        # print(f'After norm: {node_features.shape}')
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
            # print(f'After out_dropout: {node_features.shape}')

        # outputs
        # [num_atoms*batch_size, irreps_dim] -> [num_atoms*batch_size, 1]
        if self.use_attn_head:
            outputs = self.head(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )
        else:
            outputs = self.head(node_features)
        outputs = self.scale_scatter(outputs, batch, dim=0)

        if self.scale is not None:
            outputs = self.scale * outputs

        energy = outputs
        # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L321-L328
        forces = -1 * (
            torch.autograd.grad(
                energy,
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
            )[0]
        )

        return energy, forces




@register_model
def deq_graph_attention_transformer_nonlinear_l2_md17(
    irreps_in,
    radius,
    num_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    irreps_node_embedding="128x0e+64x1e+32x2e",
    num_layers=6,
    irreps_node_attr="1x0e",
    irreps_sh="1x0e+1x1e+1x2e",
    fc_neurons=[64, 64],
    irreps_feature="512x0e",
    irreps_head="32x0e+16x1e+8x2e",
    num_heads=4,
    irreps_pre_attn=None,
    rescale_degree=False,
    nonlinear_message=True,
    irreps_mlp_mid="384x0e+192x1e+96x2e",
    norm_layer="layer",
    alpha_drop=0.2,
    proj_drop=0.0,
    out_drop=0.0,
    drop_path_rate=0.0,
    scale=None,
    # DEQ specific
    deq_kwargs={},
    torchdeq_norm=omegaconf.OmegaConf.create({'norm_type': 'weight_norm'}),
    init_z_from_enc=False,  # True=V1, False=V2
    irreps_node_embedding_injection="64x0e+32x1e+16x2e",
    **kwargs,
):
    model = DEQGraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding=irreps_node_embedding,
        num_layers=num_layers,
        irreps_node_attr=irreps_node_attr,
        irreps_sh=irreps_sh,
        max_radius=radius,
        number_of_basis=num_basis,
        fc_neurons=fc_neurons,
        irreps_feature=irreps_feature,
        irreps_head=irreps_head,
        num_heads=num_heads,
        irreps_pre_attn=irreps_pre_attn,
        rescale_degree=rescale_degree,
        nonlinear_message=nonlinear_message,
        irreps_mlp_mid=irreps_mlp_mid,
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
        irreps_node_embedding_injection=irreps_node_embedding_injection,
    )
    print(f" ! Ignore passed kwargs: {kwargs}")
    return model
