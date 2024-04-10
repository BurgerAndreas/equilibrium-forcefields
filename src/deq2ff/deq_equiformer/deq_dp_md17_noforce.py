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

import equiformer.datasets.pyg.md17_backup as md17_dataset

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

from .deq_dp_md17 import DEQDotProductAttentionTransformerMD17


class DEQDotProductAttentionTransformerMD17NoForce(
    DEQDotProductAttentionTransformerMD17
):

    # @torch.enable_grad() # NOFORCE
    def encode(self, node_atom, pos, batch):
        """Encode the input graph into node features and edge features.
        Input injection.
        Basically the first third of DotProductAttentionTransformerMD17.forward()
        """
        # pos = pos.requires_grad_(True) # NOFORCE

        edge_src, edge_dst = radius_graph(
            pos, r=self.max_radius, batch=batch, max_num_neighbors=1000
        )
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization="component",
        )

        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        edge_degree_embedding = self.edge_deg_embed(
            atom_embedding, edge_sh, edge_length_embedding, edge_src, edge_dst, batch
        )
        # addition is fine
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        # atom_embedding torch.Size([168, 480])
        # edge_degree_embedding torch.Size([168, 480])

        return (
            node_features,
            node_attr,
            edge_src,
            edge_dst,
            edge_sh,
            edge_length_embedding,
            batch,
            pos,
        )

    # @torch.enable_grad() # NOFORCE
    def deq_implicit_layer(
        self,
        node_features,  # [num_atoms*batch_size, 480]
        u,
        node_features_injection=None,
    ):
        """
        Same as deq_implicit_layer but with input injection summarized in u.
        Basically the middle third of DotProductAttentionTransformerMD17.forward()
        """
        node_attr, edge_src, edge_dst, edge_sh, edge_length_embedding, batch = u

        if self.input_injection == False:  # V1
            node_features_in = node_features
        else:
            # inject node_features_injection
            # TODO does not seem right
            node_features_in = torch.cat(
                [node_features, node_features_injection], dim=1
            )

        # print("node_features.shape", node_features.shape)
        for blknum, blk in enumerate(self.blocks):
            node_features = blk(
                node_input=node_features_in,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )
        return node_features

    # @torch.enable_grad() # NOFORCE
    def decode(self, node_features, u, batch, pos):
        """Decode the node features into energy and forces (scalars).
        Basically the last third of DotProductAttentionTransformerMD17.forward()
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

        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
        outputs = self.head(node_features)
        outputs = self.scale_scatter(outputs, batch, dim=0)

        if self.scale is not None:
            outputs = self.scale * outputs

        energy = outputs

        # NOFORCE
        # # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L321-L328
        # forces = -1 * (
        #     torch.autograd.grad(
        #         energy,
        #         pos,
        #         grad_outputs=torch.ones_like(energy),
        #         create_graph=True,
        #     )[0]
        # )
        forces = None

        return energy, forces


# from .deq_dp_attention_transformer_md17 import deq_dot_product_attention_transformer_exp_l2_md17


@register_model
def deq_dot_product_attention_transformer_exp_l2_md17_noforce(
    irreps_in,
    radius,
    num_layers=6,
    number_of_basis=128,
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
    torchdeq_norm=True,
    input_injection="first_layer",
    z0="zero",
    **kwargs,
):
    # dot_product_attention_transformer_exp_l2_md17
    model = DEQDotProductAttentionTransformerMD17NoForce(
        irreps_in=irreps_in,
        num_layers=num_layers,
        irreps_node_attr=irreps_node_attr,
        max_radius=radius,
        number_of_basis=number_of_basis,
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
        input_injection=input_injection,
        z0=z0,
    )
    print(f"! Ignoring kwargs: {kwargs}")
    return model
