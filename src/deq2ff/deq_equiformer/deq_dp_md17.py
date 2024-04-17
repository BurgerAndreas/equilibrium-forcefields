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

import deq2ff.logging_utils_deq as logging_utils_deq
from deq2ff.deq_equiformer.deq_equiformer_base import EquiformerDEQBase

from deq2ff.deq_equiformer.deq_decprojhead_dp_md17 import (
    FFProjection,
    FFProjectionNorm,
    FFResidualFCTPProjection,
    FCTPProjection,
    FCTPProjectionNorm,
)

from deq2ff.deq_equiformer.deq_dp_minimal import (
    DPA, DPANorm, FF, FFNorm, FFResidual, FFNormResidual, DPAFFNorm
)

class DEQDotProductAttentionTransformerMD17(torch.nn.Module, EquiformerDEQBase):
    """
    Modified from equiformer.nets.dp_attention_transformer_md17.DotProductAttentionTransformerMD17
    """

    def __init__(
        self,
        # original
        irreps_in="64x0e",
        # dim: 128*1 + 64*3 + 32*5 = 480
        irreps_node_embedding="128x0e+64x1e+32x2e",
        irreps_feature="512x0e",  # scalar output features
        num_layers=6,
        irreps_node_attr="1x0e",
        irreps_sh="1x0e+1x1e+1x2e",
        max_radius=5.0,
        number_of_basis=128,
        basis_type="gaussian",
        fc_neurons=[64, 64],
        irreps_head="32x0e+16x1o+8x2e",
        num_heads=4,
        irreps_pre_attn=None,
        rescale_degree=False,
        nonlinear_message=False,
        irreps_mlp_mid="128x0e+64x1e+32x2e",
        norm_layer="layer",
        # regularization
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        task_mean=None,
        task_std=None,
        # scale the final output by this number
        scale: float = None,
        atomref=None,
        use_attn_head=False,
        **kwargs,
    ):
        super().__init__()

        kwargs = self._set_deq_vars(irreps_node_embedding, **kwargs)

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = task_mean
        self.task_std = task_std
        self.scale = scale
        self.register_buffer("atomref", atomref)

        self.irreps_node_attr = o3.Irreps(irreps_node_attr)
        # self.irreps_node_input = o3.Irreps(irreps_in)
        # self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        self.irreps_feature = o3.Irreps(irreps_feature)
        self.num_layers = num_layers
        self.irreps_edge_attr = (
            o3.Irreps(irreps_sh)
            if irreps_sh is not None
            else o3.Irreps.spherical_harmonics(self.lmax)
        )
        self.fc_neurons = [self.number_of_basis] + fc_neurons
        self.irreps_head = o3.Irreps(irreps_head)
        self.num_heads = num_heads
        self.irreps_pre_attn = irreps_pre_attn
        self.rescale_degree = rescale_degree
        self.nonlinear_message = nonlinear_message
        self.irreps_mlp_mid = o3.Irreps(irreps_mlp_mid)

        # Encoder
        self.atom_embed = NodeEmbeddingNetwork(
            # self.irreps_node_embedding,
            self.irreps_node_injection,
            _MAX_ATOM_TYPE,
        )
        self.basis_type = basis_type
        if self.basis_type == "gaussian":
            self.rbf = GaussianRadialBasisLayer(
                self.number_of_basis, cutoff=self.max_radius
            )
        elif self.basis_type == "bessel":
            self.rbf = RadialBasis(
                self.number_of_basis,
                cutoff=self.max_radius,
                rbf={"name": "spherical_bessel"},
            )
        elif self.basis_type == "exp":
            self.rbf = ExpNormalSmearing(
                cutoff_lower=0.0,
                cutoff_upper=self.max_radius,
                num_rbf=self.number_of_basis,
                trainable=False,
            )
        else:
            raise ValueError
        self.edge_deg_embed = EdgeDegreeEmbeddingNetwork(
            # self.irreps_node_embedding,
            self.irreps_node_injection,
            self.irreps_edge_attr,
            self.fc_neurons,
            _AVG_DEGREE,
        )

        # Implicit layers
        self.blocks = torch.nn.ModuleList()
        self.build_blocks()

        # Decoder
        # Layer Norm
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)
        # Output head
        self.use_attn_head = use_attn_head
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
        else:
            self.head = torch.nn.Sequential(
                LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
                # Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
                Activation(
                    self.irreps_feature, acts=[eval(f"torch.nn.{self.activation}()")]
                ),
                LinearRS(self.irreps_feature, o3.Irreps("1x0e"), rescale=_RESCALE),
            )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)
        self.blocks.apply(self._init_weights_blocks)

        # self._init_decoder_proj_final_layer()
        kwargs = self._init_deq(**kwargs)
        print(f"Ignoring kwargs: {kwargs}")

        #################################################################

    def build_blocks(self):
        """N blocks of: Layer Norm 1 -> DotProductAttention -> Layer Norm 2 -> FeedForwardNetwork
        Last block outputs scalars (l0) only.
        """
        for i in range(self.num_layers):
            block_type = "DPTransBlock"
            if self.deq_block is not None:
                block_type = self.deq_block
            if i == (self.num_layers - 1):
                # last block
                # last block outputs scalars only (l0 only, no higher l's)
                # irreps_node_embedding -> irreps_feature
                # "128x0e+64x1e+32x2e" -> "512x0e"
                # as of now we do not concat the node_features_input_injection
                # onto the node_features for the decoder
                irreps_node_input = self.irreps_node_embedding
                irreps_block_output = self.irreps_feature
                if self.dec_proj is not None:
                    block_type = self.dec_proj
            else:
                # first and middle layer: input injection
                irreps_node_input = self.irreps_node_z
                irreps_block_output = self.irreps_node_embedding

            if self.input_injection == "first_layer":
                if i == 0:
                    # first layer unchanged
                    pass
                elif i >= (self.num_layers - 1):
                    # last layer unchanged
                    pass
                else:
                    # middle layers: no input injection
                    irreps_node_input = self.irreps_node_embedding
                    irreps_block_output = self.irreps_node_embedding

            # Layer Norm 1 -> DotProductAttention -> Layer Norm 2 -> FeedForwardNetwork
            # extra stuff (= everything except node_features) is used for KV in DotProductAttention
            # blk = DPTransBlock(
            blk = eval(block_type)(
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
                affine_ln=self.affine_ln, 
            )
            if i != (self.num_layers - 1):
                self.blocks.append(blk)
            else:
                self.final_block = blk
            print(f'Initialized block {i} of type {block_type}.')
        print(f"\nInitialized {len(self.blocks)} blocks.")
    
    def custom_weight_init(self, m, val, ptype="weight"):
        if isinstance(val, float) or isinstance(val, int):
            if ptype == "parameterlist":
                for param in m:
                    torch.nn.init.constant_(param, val=float(val))
            else:
                torch.nn.init.constant_(eval(f'm.{ptype}'), val=float(val))
        elif 'normal' in val:
            mean = val.split("_")[1]
            std = val.split("_")[2]
            if ptype == "parameterlist":
                for param in m:
                    torch.nn.init.normal_(param, mean=float(mean), std=float(std))
            else:
                torch.nn.init.normal_(eval(f'm.{ptype}'), mean=float(mean), std=float(std))
        elif 'uniform' in val:
            a = val.split("_")[1]
            b = val.split("_")[2]
            if ptype == "parameterlist":
                for param in m:
                    torch.nn.init.uniform_(param, a=float(a), b=float(b))
            else:
                torch.nn.init.uniform_(eval(f'm.{ptype}'), a=float(a), b=float(b))
        elif val in ['torch', 'equiformer']:
            pass
        else:
            raise ValueError(f"Invalid custom_weight_init: {val}")

    def _init_weights_base(self, m, weight_init):
        """
        EquivariantLayerNormV2 are initialized to weight=0, bias=1.
        ParameterList are initialized to 0.
        """
        # torch.nn.init.normal_(tensor, mean=0.0, std=1.0)
        # kaiman
        # torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        # torch.nn.init.zeros_(self.fc1.bias)
        if isinstance(m, torch.nn.Linear):
            # initialized uniformly :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, 
            # where :math:`k = \frac{1}{\text{in\_features}}`
            # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            # weight and bias kaiming_uniform
            self.custom_weight_init(m, weight_init['Linear_w'], ptype="weight")
            
            if weight_init['Linear_b'] == "equiformer":
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            else:
                self.custom_weight_init(m, weight_init['Linear_b'], ptype="bias")

        elif isinstance(m, torch.nn.LayerNorm):
            if weight_init['LayerNorm_w'] == "equiformer":
                torch.nn.init.constant_(m.weight, 1.0)
            else:
                self.custom_weight_init(m, weight_init['LayerNorm_w'], ptype="weight")

            if weight_init['LayerNorm_b'] == "equiformer":
                torch.nn.init.constant_(m.bias, 0)
            else:
                self.custom_weight_init(m, weight_init['LayerNorm_b'], ptype="bias")

        elif isinstance(m, EquivariantLayerNormV2):
            # https://github.com/atomicarchitects/equiformer_v2/blob/main/nets/equiformer_v2/equiformer_v2_oc20.py#L489
            if m.affine:
                # weight=0, bias=1
                # torch.nn.init.constant_(m.affine_weight, 0)
                # torch.nn.init.constant_(m.affine_bias, 1)
                # if self.weight_init == 'normal':
                #         std = 1 / math.sqrt(m.in_features)
                #         torch.nn.init.normal_(m.weight, 0, std)
                self.custom_weight_init(m, weight_init['EquivariantLayerNormV2_w'], ptype="affine_weight")
                self.custom_weight_init(m, weight_init['EquivariantLayerNormV2_b'], ptype="affine_bias")
        
        elif isinstance(m, torch.nn.ParameterList):
            # for param in m:
            #     torch.nn.init.zeros_(param)
            self.custom_weight_init(m, weight_init['ParameterList'], ptype="parameterlist")

    
    def _init_weights(self, m):
        self._init_weights_base(m, weight_init=self.weight_init)
    
    def _init_weights_blocks(self, m):
        self._init_weights_base(m, weight_init=self.weight_init_blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        """?"""
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormV2)
                or isinstance(module, EquivariantInstanceNorm)
                or isinstance(module, EquivariantGraphNorm)
                or isinstance(module, GaussianRadialBasisLayer)
                or isinstance(module, RadialBasis)
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, torch.nn.Linear)
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)

    # def _init_z(self):
    #     return torch.zeros(1, self.d_hidden)

    def _init_z(self, batch_size, dim):
        """Initializes fixed-point for DEQ
        shape: [num_atoms * batch_size, irreps_dim]
        irreps_dim = a*1 + b*3 + c*5
        """
        requires_grad = self.z0_requires_grad # TODO
        if self.z0 == "zero":
            return torch.zeros(
                [batch_size, dim],
                device=self.device,
                requires_grad=requires_grad,
            )
        elif self.z0 == "one":
            return torch.ones(
                [batch_size, dim],
                device=self.device,
                requires_grad=requires_grad,
            )
        elif self.z0 == "uniform":
            return torch.rand(
                [batch_size, dim],
                device=self.device,
                requires_grad=requires_grad,
            )
        elif "normal" in self.z0:
            # normal_mean_std = normal_0.0_0.5
            mean = self.z0.split("_")[1]
            std = self.z0.split("_")[2]
            return torch.normal(
                mean=float(mean), std=float(std), 
                size=[batch_size, dim], 
                device=self.device,
                requires_grad=requires_grad,
            )
        else:
            raise ValueError(f"Invalid z0: {self.z0}")

    @torch.enable_grad()
    def encode(self, node_atom, pos, batch):
        """Encode the input graph into node features and edge features.
        Input injection.
        Basically the first third of DotProductAttentionTransformerMD17.forward()
        """

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

        # atom_embedding [num_atoms*batch_size, irreps_dim]
        # edge_degree_embedding [num_atoms*batch_size, irreps_dim]

        # requires_grad: node_features, edge_sh, edge_length_embedding
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
        node_features_injection,
    ):
        """
        Same as deq_implicit_layer but with input injection summarized in u.
        Basically the middle third of DotProductAttentionTransformerMD17.forward()
        """

        # [num_atoms*batch_size, 480]
        if self.input_injection == False:
            # no injection, injection becomes the initial input
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
        elif self.input_injection == "every_layer":
            # input injection at every layer
            for blknum, blk in enumerate(self.blocks):
                node_features = torch.cat(
                    [node_features, node_features_injection], dim=1
                )
                node_features = blk(
                    node_input=node_features,
                    node_attr=node_attr,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_attr=edge_sh,
                    edge_scalars=edge_length_embedding,
                    batch=batch,
                )
        elif self.input_injection == "first_layer":
            # input injection only at the first layer
            # node features does not require_grad until concat with injection
            node_features = torch.cat([node_features, node_features_injection], dim=1)
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

        elif self.input_injection == "legacy":
            # print("!"*60, "\nDepricated: Legacy input injection")
            node_features_in = torch.cat(
                [node_features, node_features_injection], dim=1
            )
            # # print("node_features.shape", node_features.shape)
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

        else:
            raise ValueError(f"Invalid input_injection: {self.input_injection}")

        return node_features

    @torch.enable_grad()
    def decode(
        self,
        node_features,
        node_attr,
        edge_src,
        edge_dst,
        edge_sh,
        edge_length_embedding,
        batch,
        pos,
    ):
        """Decode the node features into energy and forces (scalars).
        Basically the last third of DotProductAttentionTransformerMD17.forward()
        """

        node_features = self.final_block(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,  # requires_grad
            edge_scalars=edge_length_embedding,  # requires_grad
            batch=batch,
        )

        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)

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
        # shape: [num_atoms*batch_size, 3]
        forces = -1 * (
            torch.autograd.grad(
                energy,
                # diff with respect to pos
                # if you get 'One of the differentiated Tensors appears to not have been used in the graph'
                # then because pos is not 'used' to calculate the energy
                pos,
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
                # allow_unused=True, # TODO
            )[0]
        )

        return energy, forces

    def dummy_forward_for_logging(self, node_atom, pos, batch, **kwargs):
        """Return dictionary of shapes."""

        (
            node_features_injection,
            node_attr,
            edge_src,
            edge_dst,
            edge_sh,
            edge_length_embedding,
            batch,
            pos,
        ) = self.encode(node_atom=node_atom, pos=pos, batch=batch)

        if self.input_injection == False:
            node_features = node_features_injection
        else:
            node_features = self._init_z(batch_size=node_features_injection.shape[0], dim=self.irreps_node_embedding.dim)

        # f = lambda z: self.mfn_forward(z, u)
        f = lambda node_features: self.deq_implicit_layer(
            node_features,
            node_attr,
            edge_src,
            edge_dst,
            edge_sh,
            edge_length_embedding,
            batch,
            node_features_injection,
        )

        logs = {
            "NumNodes": node_features_injection.shape[0],
            "NumEdges": edge_src.shape[0],
            "DimInputInjection": node_features_injection.shape[1],
            "DimFixedPoint": node_features.shape[1],
        }
        return logs

    def forward(
        self,
        node_atom,
        pos,
        batch,
        z=None,
        step=None,
        datasplit=None,
        return_fixedpoint=False,
        fixedpoint=None,
    ):
        """Forward pass of the DEQ model."""
        pos = pos.requires_grad_(True)

        # encode
        # u = self.encode(x)
        (
            node_features_injection,
            node_attr,
            edge_src,
            edge_dst,
            edge_sh,
            edge_length_embedding,
            batch,
            pos,
        ) = self.encode(node_atom=node_atom, pos=pos, batch=batch)

        if self.input_injection == False:
            node_features = node_features_injection
        else:
            node_features = self._init_z(batch_size=node_features_injection.shape[0], dim=self.irreps_node_embedding.dim)

        reuse = True
        if fixedpoint is None:
            # z = torch.zeros(x.shape[0], self.d_hidden).to(x)
            reuse = False
        else:
            node_features = fixedpoint

        # debugging
        if self.skip_implicit_layer:
            node_features = self._init_z(batch_size=node_features_injection.shape[0], dim=self.irreps_node_embedding.dim)
            z_pred = [node_features]
            # print("! Skipping implicit layer")
        else:
            reset_norm(self.blocks)

            # f = lambda z: self.mfn_forward(z, u)
            f = lambda node_features: self.deq_implicit_layer(
                node_features,
                node_attr,
                edge_src,
                edge_dst,
                edge_sh,
                edge_length_embedding,
                batch,
                node_features_injection,
            )

            # z: list[torch.tensor shape [42, 480]]
            solver_kwargs = {"f_max_iter": 0} if (reuse and self.limit_f_max_iter_fpreuse) else {}
            # returns the sampled fixed point trajectory (tracked gradients)
            # z_pred, info = self.deq(f, z, solver_kwargs=solver_kwargs)
            z_pred, info = self.deq(f, node_features, solver_kwargs=solver_kwargs)

            if step is not None:
                # log fixed-point trajectory
                _data = logging_utils_deq.log_fixed_point_error(
                    info,
                    step,
                    datasplit,
                    self.fp_error_traj[datasplit],
                    log_fp_error_traj=self.log_fp_error_traj,
                )
                if _data is not None:
                    self.fp_error_traj[datasplit] = _data
                # log the final fixed-point
                logging_utils_deq.log_fixed_point_norm(z_pred, step, datasplit)
                # log the input injection (output of encoder)
                logging_utils_deq.log_fixed_point_norm(node_features_injection, step, datasplit, name="emb")

        # decode
        # outputs: list[Tuple(energy: torch.tensor [2, 1], force: torch.tensor [42, 3])]
        # outputs = [self.out(z) for z in z_pred]

        # outputs = [self.decode(node_features=z, u=u, batch=batch, pos=pos) for z in z_pred]
        # energy = outputs[-1][0]
        # force = outputs[-1][1]

        if not z_pred[-1].requires_grad:
            print("!!!", f"z_pred[-1] node_features.requires_grad: {z_pred[-1].requires_grad} (datasplit: {datasplit})"            )

        energy, force = self.decode(
            node_features=z_pred[-1],
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_sh=edge_sh,
            edge_length_embedding=edge_length_embedding,
            batch=batch,
            pos=pos,
        )

        # return outputs, z_pred[-1]
        if return_fixedpoint:
            # z_pred = sampled fixed point trajectory (tracked gradients)
            return energy, force, z_pred[-1].detach().clone()
        return energy, force


@register_model
def deq_dot_product_attention_transformer_exp_l2_md17(
    # irreps_in,
    # radius,
    # num_layers=6,
    # number_of_basis=128,
    # atomref=None,
    # task_mean=None,
    # task_std=None,
    # irreps_node_attr="1x0e",
    # basis_type="exp",
    # # most import for parameter count?
    # fc_neurons=[64, 64],
    # irreps_node_embedding_injection="64x0e+32x1e+16x2e",
    # irreps_node_embedding="128x0e+64x1e+32x2e",
    # irreps_feature="512x0e",  # scalars only
    # irreps_sh="1x0e+1x1e+1x2e",
    # irreps_head="32x0e+16x1e+8x2e",
    # num_heads=4,
    # irreps_mlp_mid="384x0e+192x1e+96x2e",
    # #
    # irreps_pre_attn=None,
    # rescale_degree=False,
    # nonlinear_message=False,
    # norm_layer="layer",
    # # regularization
    # alpha_drop=0.0,
    # proj_drop=0.0,
    # out_drop=0.0,
    # drop_path_rate=0.0,
    # scale=None,
    # # DEQ specific
    # deq_kwargs={},
    # torchdeq_norm=omegaconf.OmegaConf.create({"norm_type": "weight_norm"}),
    # input_injection="first_layer",
    # z0="zero",
    # log_fp_error_traj=False,
    # dp_tp_path_norm="none",
    # dp_tp_irrep_norm=None, # None = 'element'
    # fc_tp_path_norm="none",
    # fc_tp_irrep_norm=None, # None = 'element'
    **kwargs,
):
    # dot_product_attention_transformer_exp_l2_md17
    model = DEQDotProductAttentionTransformerMD17(**kwargs)
    return model
