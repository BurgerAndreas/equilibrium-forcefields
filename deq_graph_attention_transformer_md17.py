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


import e3nn
from e3nn import o3
from e3nn.util.jit import compile_mode
from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

from equiformer.nets.registry import register_model


class DEQGraphAttentionTransformerMD17(nets.graph_attention_transformer_md17.GraphAttentionTransformerMD17):

    def __init__(self, deq_mode=True, **kwargs):
        super().__init__(**kwargs)

        self.deq_mode = deq_mode
        self.deq = get_deq(**kwargs)


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

        # 21, 512 -> 21, 512
        node_features = self.norm(node_features, batch=batch)
        print(f'After norm: {node_features.shape}')


        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)
            print(f'After out_dropout: {node_features.shape}')
        
        # outputs
        # 21, 512 -> 21, 1
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

# copy from deq_dp_attention_transformer_md17.py
from deq_dp_attention_transformer_md17 import DEQDotProductAttentionTransformerMD17

DEQGraphAttentionTransformerMD17.forward = DEQDotProductAttentionTransformerMD17.forward 
DEQGraphAttentionTransformerMD17.encode = DEQDotProductAttentionTransformerMD17.encode 
DEQGraphAttentionTransformerMD17.deq_implicit_layer = DEQDotProductAttentionTransformerMD17.deq_implicit_layer 

@register_model
def deq_graph_attention_transformer_nonlinear_l2_md17(
    irreps_in,
    radius,
    num_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    **kwargs
):
    model = DEQGraphAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding="128x0e+64x1e+32x2e",
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=radius,
        number_of_basis=num_basis,
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
        mean=task_mean,
        std=task_std,
        scale=None,
        atomref=atomref,
        # DEQ specific
        deq_mode=True,
    )
    return model