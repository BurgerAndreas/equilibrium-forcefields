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


from e3nn import o3
# import e3nn
# from e3nn.util.jit import compile_mode
# from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

from equiformer.nets.registry import register_model


class DEQDotProductAttentionTransformerMD17(nets.dp_attention_transformer_md17.DotProductAttentionTransformerMD17):
    """
    Modified from equiformer.nets.dp_attention_transformer_md17.DotProductAttentionTransformerMD17
    """

    def __init__(self, deq_mode=True, deq_kwargs={}, **kwargs):
        # print(f'DEQDotProductAttentionTransformerMD17 passed kwargs: {kwargs}')
        super().__init__(**kwargs)

        # implicit layer
        # self.mfn = nn.ModuleList([
        #     MFNLinear(d_hidden, d_hidden) for _ in range(n_layer)
        # ])
        # self.blocks

        #################################################################
        # DEQ specific

        self.deq_mode = deq_mode
        self.deq = get_deq(**deq_kwargs)
        # self.deq = get_deq(f_solver='broyden', f_max_iter=20, f_tol=1e-6)
        # self.register_buffer('z_aux', self._init_z())

        # from DEQ INR example
        # https://colab.research.google.com/drive/12HiUnde7qLadeZGGtt7FITnSnbUmJr-I?usp=sharing#scrollTo=RGgPMQLT6IHc
        # Later: try if this helps
        # This function automatically decorates weights in your DEQ layer
        # to have weight/spectral normalization. (for better stability)
        # Using norm_type='none' in `kwargs` can also skip it.
        # apply_norm(self.mfn, **kwargs)
        # apply_norm(self.blocks, **kwargs)

    # I don't think we need this
    # Since our z = node_features is coming from the encoder
    # def _init_z(self):
    #     return torch.zeros(1, self.d_hidden)

    @torch.enable_grad()
    def encode(self, node_atom, pos, batch):
        """Encode the input graph into node features and edge features.
        Input injection.
        Basically the first third of DotProductAttentionTransformerMD17.forward()
        """
        pos = pos.requires_grad_(True)

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
        node_features = atom_embedding + edge_degree_embedding
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

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

    @torch.enable_grad()
    def deq_implicit_layer(
        self,
        node_features,
        node_attr,
        edge_src,
        edge_dst,
        edge_sh,
        edge_length_embedding,
        batch,
    ):
        """
        Basically the middle third of DotProductAttentionTransformerMD17.forward()
        """
        for blk in self.blocks:
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )
        return node_features
    
    @torch.enable_grad()
    def deq_implicit_layer_u(
        self,
        node_features,
        u,
    ):
        """
        Same as deq_implicit_layer but with input injection summarized in u.
        Basically the middle third of DotProductAttentionTransformerMD17.forward()
        """
        node_attr, edge_src, edge_dst, edge_sh, edge_length_embedding, batch = u
        # print("node_features.shape", node_features.shape)
        for blknum, blk in enumerate(self.blocks):
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                batch=batch,
            )
            # print(f"After block {blknum} node_features.shape", node_features.shape)
        return node_features

    @torch.enable_grad()
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

    def forward(self, node_atom, pos, batch, z=None, return_grad=False):
        """Forward pass of the DEQ model."""

        # encode
        # u = self.encode(x)
        (
            node_features,
            node_attr,
            edge_src,
            edge_dst,
            edge_sh,
            edge_length_embedding,
            batch,
            pos,
        ) = self.encode(node_atom=node_atom, pos=pos, batch=batch)

        reuse = True
        if z is None:
            # z = torch.zeros(x.shape[0], self.d_hidden).to(x)
            reuse = False
        else:
            node_features = z

        # reset_norm(self.mfn)
        # f = lambda z: self.mfn_forward(z, u)
        u = (node_attr, edge_src, edge_dst, edge_sh, edge_length_embedding, batch)
        f = lambda node_features: self.deq_implicit_layer_u(node_features, u)

        # z: list[torch.tensor shape [42, 480]]
        if self.deq_mode:
            solver_kwargs = {"f_max_iter": 0} if reuse else {}
            # returns the sampled fixed point trajectory (tracked gradients)
            # z_pred, info = self.deq(f, z, solver_kwargs=solver_kwargs)
            z_pred, info = self.deq(f, node_features, solver_kwargs=solver_kwargs)

        else:
            z_pred = [f(z)]

        # decode
        # outputs: list[Tuple(energy: torch.tensor [2, 1], force: torch.tensor [42, 3])]
        # outputs = [self.out(z) for z in z_pred]
            
        # outputs = [self.decode(node_features=z, u=u, batch=batch, pos=pos) for z in z_pred]
        # energy = outputs[-1][0]
        # force = outputs[-1][1] 

        energy, force = self.decode(node_features=z_pred[-1], u=u, batch=batch, pos=pos)

        # return outputs, z_pred[-1]
        if return_grad:
            # z_pred = sampled fixed point trajectory (tracked gradients)
            return energy, force, z_pred
        return energy, force



@register_model
def deq_dot_product_attention_transformer_exp_l2_md17(
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
    irreps_node_embedding="128x0e+64x1e+32x2e",
    irreps_sh="1x0e+1x1e+1x2e",
    irreps_feature="512x0e",
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
    **kwargs
):
    # dot_product_attention_transformer_exp_l2_md17
    model = DEQDotProductAttentionTransformerMD17(
        irreps_in=irreps_in,
        num_layers=num_layers,
        irreps_node_attr=irreps_node_attr,
        max_radius=radius,
        number_of_basis=num_basis,
        basis_type=basis_type,
        # most import for parameter count?
        fc_neurons=fc_neurons,
        irreps_node_embedding=irreps_node_embedding,
        irreps_sh=irreps_sh,
        irreps_feature=irreps_feature,
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
    )
    print(f'! Ignoring kwargs: {kwargs}')
    return model

