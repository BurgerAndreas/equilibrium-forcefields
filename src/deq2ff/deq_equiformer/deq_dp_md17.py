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


class DEQDotProductAttentionTransformerMD17(torch.nn.Module):
    """
    Modified from equiformer.nets.dp_attention_transformer_md17.DotProductAttentionTransformerMD17
    """

    def __init__(
        self,
        # added
        deq_mode=True,
        torchdeq_norm=omegaconf.OmegaConf.create({'norm_type': 'weight_norm'}),
        deq_kwargs={},
        input_injection='first_layer',  # False=V1, 'first_layer'=V2
        irreps_node_embedding_injection="64x0e+32x1e+16x2e",
        z0='zero',
        # original
        irreps_in="64x0e",
        # 128*1 + 64*3 + 32*5 = 480
        irreps_node_embedding="128x0e+64x1e+32x2e",
        irreps_feature="512x0e", # scalar output features
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
        alpha_drop=0.2,
        proj_drop=0.0,
        out_drop=0.0,
        drop_path_rate=0.0,
        mean=None,
        std=None,
        scale=None,
        atomref=None,
    ):
        super().__init__()

        #################################################################
        # Added

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_injection = input_injection
        if input_injection is False:
            # V1
            # node_features are initialized as the output of the encoder
            self.irreps_node_injection = o3.Irreps(
                irreps_node_embedding
            )  # output of encoder
            self.irreps_node_z = o3.Irreps(irreps_node_embedding)  # input to block
            self.irreps_node_embedding = o3.Irreps(
                irreps_node_embedding
            )  # output of block
        elif self.input_injection in ['first_layer', 'every_layer', True, 'legacy']:
            # V2
            # node features are initialized as 0
            # and the node features from the encoder are used as input injection
            # encoder = atom_embed() and edge_deg_embed()
            # both encoder shapes are defined by irreps_node_embedding
            # input to self.blocks is the concat of node_input and node_injection
            self.irreps_node_injection = o3.Irreps(
                irreps_node_embedding_injection
            )  # output of encoder
            self.irreps_node_embedding = o3.Irreps(
                irreps_node_embedding
            )  # output of block
            # "128x0e+64x1e+32x2e" + "64x0e+32x1e+16x2e"
            # 128x0e+64x1e+32x2e+64x0e+32x1e+16x2e
            irreps_node_z = self.irreps_node_embedding + self.irreps_node_injection
            irreps_node_z.simplify()
            self.irreps_node_z = o3.Irreps(irreps_node_z).simplify()  # input to block
        else:
            raise ValueError(f"Invalid input_injection: {input_injection}")
        #################################################################

        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.alpha_drop = alpha_drop
        self.proj_drop = proj_drop
        self.out_drop = out_drop
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.task_mean = mean
        self.task_std = std
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
        self.head = torch.nn.Sequential(
            LinearRS(self.irreps_feature, self.irreps_feature, rescale=_RESCALE),
            Activation(self.irreps_feature, acts=[torch.nn.SiLU()]),
            LinearRS(self.irreps_feature, o3.Irreps("1x0e"), rescale=_RESCALE),
        )
        self.scale_scatter = ScaledScatter(_AVG_NUM_NODES)

        self.apply(self._init_weights)
        self.dec_proj = None

        #################################################################
        # DEQ specific

        self.deq_mode = deq_mode
        self.deq = get_deq(**deq_kwargs)
        # self.deq = get_deq(f_solver='broyden', f_max_iter=20, f_tol=1e-6)
        # self.register_buffer('z_aux', self._init_z())

        # from DEQ INR example
        # https://colab.research.google.com/drive/12HiUnde7qLadeZGGtt7FITnSnbUmJr-I?usp=sharing#scrollTo=RGgPMQLT6IHc
        # https://github.com/locuslab/torchdeq/blob/main/deq-zoo/ignn/graphclassification/layers.py
        # This function automatically decorates weights in your DEQ layer
        # to have weight/spectral normalization. (for better stability)
        # Using norm_type='none' in `kwargs` can also skip it.
        if torchdeq_norm.norm_type not in [None, 'none', False]:
            apply_norm(self.blocks, **torchdeq_norm)
            # register_norm_module(DEQDotProductAttentionTransformerMD17, 'spectral_norm', names=['blocks'], dims=[0])
        #################################################################

    def build_blocks(self):
        """N blocks of: Layer Norm 1 -> DotProductAttention -> Layer Norm 2 -> FeedForwardNetwork
        Last block outputs scalars (l0) only.
        """
        for i in range(self.num_layers):
            # last layer is different which will screw up DEQ
            # last block outputs scalars only (l0 only, no higher l's)
            # irreps_node_embedding -> irreps_feature
            # "128x0e+64x1e+32x2e" -> "512x0e"
            if i >= (self.num_layers - 1):
                # last block
                # as of now we do not concat the node_features_input_injection 
                # onto the node_features for the decoder
                irreps_node_input = self.irreps_node_embedding
                irreps_block_output = self.irreps_feature
            else:
                irreps_node_input = self.irreps_node_z
                irreps_block_output = self.irreps_node_embedding

            if self.input_injection == 'first_layer':
                if i == 0:
                    pass
                    # first layer unchanged
                elif i >= (self.num_layers - 1):
                    # last layer unchanged
                    pass
                    # irreps_node_input = self.irreps_node_embedding
                    # irreps_block_output = self.irreps_feature
                else:
                    # no input injection
                    irreps_node_input = self.irreps_node_embedding
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
            if i != (self.num_layers - 1):
                self.blocks.append(blk)
            else:
                self.final_block = blk
        print(f'\nInitialized {len(self.blocks)} blocks of `DPTransBlock`.')

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
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

    def _init_z(self, node_features_injection):
        """Initializes fixed-point for DEQ
        shape: [num_atoms * batch_size, irreps_dim]
        irreps_dim = a*1 + b*3 + c*5 
        """
        # return torch.zeros(1, self.irreps_feature.dim)
        if self.z0 == 'zero':
            return torch.zeros([
                node_features_injection.shape[0],
                self.irreps_node_embedding.dim,
            ], device=self.device)
        elif self.z0 == 'rand':
            return torch.randn([
                node_features_injection.shape[0],
                self.irreps_node_embedding.dim,
            ], device=self.device)
        elif self.z0 == 'one':
            return torch.ones([
                node_features_injection.shape[0],
                self.irreps_node_embedding.dim,
            ], device=self.device)
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

        # atom_embedding torch.Size([168, 480])
        # edge_degree_embedding torch.Size([168, 480])

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
        node_attr, edge_src, edge_dst, edge_sh, edge_length_embedding, batch,
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
        elif self.input_injection == 'every_layer':
            # input injection at every layer
            for blknum, blk in enumerate(self.blocks):
                node_features = torch.cat([node_features, node_features_injection], dim=1)
                node_features = blk(
                    node_input=node_features,
                    node_attr=node_attr,
                    edge_src=edge_src,
                    edge_dst=edge_dst,
                    edge_attr=edge_sh,
                    edge_scalars=edge_length_embedding,
                    batch=batch,
                )
        elif self.input_injection == 'first_layer':
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
        
        elif self.input_injection == 'legacy':
            # print("!"*60, "\nDepricated: Legacy input injection")
            node_features_in = torch.cat([node_features, node_features_injection], dim=1)
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
    
        # if self.input_injection == True:
        #     node_features_in = node_features
        # else:
        #     # inject node_features_injection
        #     # TODO does not seem right
        #     # print("cat node_features.shape", node_features.shape)
        #     node_features_in = torch.cat([node_features, node_features_injection], dim=1)

        # # print("node_features.shape", node_features.shape)
        # for blknum, blk in enumerate(self.blocks):
        #     node_features = blk(
        #         node_input=node_features_in,
        #         node_attr=node_attr,
        #         edge_src=edge_src,
        #         edge_dst=edge_dst,
        #         edge_attr=edge_sh,
        #         edge_scalars=edge_length_embedding,
        #         batch=batch,
        #     )
        # return node_features

    @torch.enable_grad()
    def decode(self, node_features, node_attr, edge_src, edge_dst, edge_sh, edge_length_embedding, batch, pos, datasplit=None):
        """Decode the node features into energy and forces (scalars).
        Basically the last third of DotProductAttentionTransformerMD17.forward()
        """
        # TODO: 
        # if model.eval() -> node_features.requires_grad is False
        # if also using decprojhead -> autograd error
        # because if final_block does not use DotProductAttention, edge_sh and edge_length_embedding are not used
        # but why are they used in prior blocks if model.eval()?

        node_features = self.final_block(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh, # requires_grad
            edge_scalars=edge_length_embedding, # requires_grad
            batch=batch,
        )

        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)

        # outputs
            # [num_atoms*batch_size, irreps_dim] -> [num_atoms*batch_size, 1]
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
            node_features = self._init_z(
                node_features_injection
            ) 

        reuse = True
        if fixedpoint is None:
            # z = torch.zeros(x.shape[0], self.d_hidden).to(x)
            reuse = False
        else:
            node_features = fixedpoint

        reset_norm(self.blocks)

        # f = lambda z: self.mfn_forward(z, u)
        f = lambda node_features: self.deq_implicit_layer(
            node_features, node_attr, edge_src, edge_dst, edge_sh, edge_length_embedding, batch, node_features_injection
        )

        # z: list[torch.tensor shape [42, 480]]
        if self.deq_mode:
            solver_kwargs = {"f_max_iter": 0} if reuse else {}
            # returns the sampled fixed point trajectory (tracked gradients)
            # z_pred, info = self.deq(f, z, solver_kwargs=solver_kwargs)
            z_pred, info = self.deq(f, node_features, solver_kwargs=solver_kwargs)
            # TODO deq() does not set z.requires_grad_() by default
            # which leads to no gradients for z in model.eval()
            # ift=True, hook_ift=True does
            # https://github.com/locuslab/torchdeq/blob/4f6bd5fa66dd991cad74fcc847c88061764cf8db/torchdeq/grad.py#L185

        else:
            z_pred = [f(z)]
            raise ValueError('DEQ mode must be True')

        if step is not None:
            deq_utils.log_fixed_point_error(info, step, datasplit)
            deq_utils.log_fixed_point_norm(z_pred, step, datasplit)

        # decode
        # outputs: list[Tuple(energy: torch.tensor [2, 1], force: torch.tensor [42, 3])]
        # outputs = [self.out(z) for z in z_pred]

        # outputs = [self.decode(node_features=z, u=u, batch=batch, pos=pos) for z in z_pred]
        # energy = outputs[-1][0]
        # force = outputs[-1][1]
            
        if not z_pred[-1].requires_grad:
            print('!'*60)
            print(f'before decode: z_pred[-1] node_features.requires_grad: {z_pred[-1].requires_grad}', flush=True)
            print(f'datasplit: {datasplit}', flush=True)
            print('!'*60)
        
        energy, force = self.decode(
            node_features=z_pred[-1], node_attr=node_attr, edge_src=edge_src, edge_dst=edge_dst, 
            edge_sh=edge_sh, edge_length_embedding=edge_length_embedding, batch=batch, pos=pos, 
            datasplit=datasplit
        )

        # return outputs, z_pred[-1]
        if return_fixedpoint:
            # z_pred = sampled fixed point trajectory (tracked gradients)
            return energy, force, z_pred[-1].detach().clone()
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
    # DEQ specific
    deq_kwargs={},
    torchdeq_norm=omegaconf.OmegaConf.create({'norm_type': 'weight_norm'}),
    input_injection='first_layer',
    z0='zero',
    **kwargs,
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
