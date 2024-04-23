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


from equiformer.nets.registry import register_model

from equiformer.nets.graph_attention_transformer import TransBlock


from deq2ff.deq_equiformer.deq_dp_md17 import DEQDotProductAttentionTransformerMD17


class DEQGraphAttentionTransformerMD17(DEQDotProductAttentionTransformerMD17):
    """Graph attention = MLP attention + non-linear message passing.
    Modified from equiformer.nets.graph_attention_transformer_md17.GraphAttentionTransformerMD17.

    Much slower than DotProductAttention, and only gives improvements on OC20.
    QM9 and MD17 are too easy.

    Only difference to between GraphAttention and DotProductAttention:
    - blocks contain TransBlock instead of DPTransBlock
    """

    def build_blocks(self):
        for i in range(self.num_layers):
            if i >= (self.num_layers - 1):
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
                    pass
                    # first layer unchanged
                elif i >= (self.num_layers - 1):
                    # last layer unchanged
                    pass
                else:
                    # middle layers: no input injection
                    irreps_node_input = self.irreps_node_embedding
                    irreps_block_output = self.irreps_node_embedding

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
                # added
                tp_path_norm=self.tp_path_norm,
                tp_irrep_norm=self.tp_irrep_norm,
                activation=self.activation,
            )
            if i != (self.num_layers - 1):
                self.blocks.append(blk)
            else:
                self.final_block = blk

        print(f"\nInitialized {len(self.blocks)} blocks of `TransBlock`.")


# Dummies for all the different versions of the model
# different argument defaults got moved to hydra configs


@register_model
def deq_graph_attention_transformer_l2_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_l2_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_l2_e3_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_bessel_l2_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_exp_l2_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_exp_l3_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_attn_exp_l3_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_exp_l3_e3_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_bessel_l3_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)


@register_model
def deq_graph_attention_transformer_nonlinear_bessel_l3_e3_md17(**kwargs):
    return DEQGraphAttentionTransformerMD17(**kwargs)
