import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3

# import e3nn
# from e3nn.util.jit import compile_mode
# from e3nn.nn.models.v2106.gate_points_message_passing import tp_path_exists

import torch_geometric
import math

from .registry import register_model
from .instance_norm import EquivariantInstanceNorm
from .graph_norm import EquivariantGraphNorm
from .layer_norm import EquivariantLayerNormV2
from .radial_func import RadialProfile
from .tensor_product_rescale import (
    TensorProductRescale,
    LinearRS,
    FullyConnectedTensorProductRescale,
    irreps2gate,
)
from .fast_activation import Activation, Gate
from .drop import EquivariantDropout, EquivariantScalarsDropout, GraphDropPath

from .gaussian_rbf import GaussianRadialBasisLayer

# for bessel radial basis
from ocpmodels.models.gemnet.layers.radial_basis import RadialBasis
from .expnorm_rbf import ExpNormalSmearing

from .graph_attention_transformer import (
    get_norm_layer,
    FullyConnectedTensorProductRescaleNorm,
    FullyConnectedTensorProductRescaleNormSwishGate,
    FullyConnectedTensorProductRescaleSwishGate,
    DepthwiseTensorProduct,
    SeparableFCTP,
    Vec2AttnHeads,
    AttnHeads2Vec,
    FeedForwardNetwork,
    GraphAttention,
    NodeEmbeddingNetwork,
    ScaledScatter,
    EdgeDegreeEmbeddingNetwork,
)
from .dp_attention_transformer import ScaleFactor, DotProductAttention, DPTransBlock

import copy
import wandb
import omegaconf

_RESCALE = True
_USE_BIAS = True

_MAX_ATOM_TYPE = 64
# Statistics of QM9 with cutoff max_radius = 5
# For simplicity, use the same statistics for MD17
_AVG_NUM_NODES = 18.03065905448718
_AVG_DEGREE = 15.57930850982666

# used to init path norm, weight init, and other tricks added for DEQ
from deq2ff.deq_equiformer.deq_equiformer_base import EquiformerDEQBase


class DotProductAttentionTransformerMD17(EquiformerDEQBase, torch.nn.Module):
    def __init__(
        self,
        irreps_in="64x0e",
        irreps_node_embedding="128x0e+64x1e+32x2e",
        irreps_feature="512x0e",
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
        # MD17 specific
        task_mean=None,
        task_std=None,
        scale=None,
        atomref=None,
        **kwargs,
    ):
        super().__init__()
        kwargs = self._set_deq_vars(
            irreps_node_embedding, irreps_feature, num_layers, **kwargs
        )
        print(f"{self.__class__.__name__} ignoring kwargs: {kwargs}")

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
        self.irreps_node_embedding = o3.Irreps(irreps_node_embedding)
        self.lmax = self.irreps_node_embedding.lmax
        # self.irreps_feature = o3.Irreps(irreps_feature)
        # self.num_layers = num_layers
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

        self.atom_embed = NodeEmbeddingNetwork(
            self.irreps_node_embedding, _MAX_ATOM_TYPE
        )
        # self.tag_embed = NodeEmbeddingNetwork(self.irreps_node_embedding, _NUM_TAGS)

        # self.attr_embed = None
        # if self.use_node_attr:
        #     self.attr_embed = NodeEmbeddingNetwork(
        #         self.irreps_node_attr, _MAX_ATOM_TYPE
        #     )

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
            self.irreps_node_embedding,
            self.irreps_edge_attr,
            self.fc_neurons,
            _AVG_DEGREE,
        )
        # self.edge_src_embed = None
        # self.edge_dst_embed = None
        # if self.use_atom_edge_attr:
        #     self.edge_src_embed = NodeEmbeddingNetwork(
        #         self.irreps_atom_edge_attr, _MAX_ATOM_TYPE
        #     )
        #     self.edge_dst_embed = NodeEmbeddingNetwork(
        #         self.irreps_atom_edge_attr, _MAX_ATOM_TYPE
        #     )

        self.blocks = torch.nn.ModuleList()
        self.build_blocks()

        # Layer Norm
        self.norm = get_norm_layer(self.norm_layer)(self.irreps_feature)
        self.out_dropout = None
        if self.out_drop != 0.0:
            self.out_dropout = EquivariantDropout(self.irreps_feature, self.out_drop)

        # Output head for energy
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
                # added
                normalization=None,
                path_normalization="none",
                activation="SiLU",
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

        # added
        if self.force_head is None:
            self.force_block = None
        else:
            # outputs = self.irreps_node_embedding # [num_atoms*batch_size, irrep_dim]
            outputs = o3.Irreps("1x1e")  # [num_atoms*batch_size, 3]
            # DPTransBlock, GraphAttention
            self.force_block = eval(self.force_head)(
                # irreps_node_input=self.irreps_node_embedding,
                irreps_node_input=self.irreps_feature,
                irreps_node_attr=self.irreps_node_attr,
                irreps_edge_attr=self.irreps_edge_attr,
                # output: which l's?
                irreps_node_output=outputs,
                fc_neurons=self.fc_neurons,
                irreps_head=self.irreps_head,
                num_heads=self.num_heads,
                irreps_pre_attn=self.irreps_pre_attn,
                rescale_degree=self.rescale_degree,
                nonlinear_message=self.nonlinear_message,
                alpha_drop=self.alpha_drop,
                proj_drop=self.proj_drop,
                activation=self.activation,
                # added
                bias=self.bias,
                # only DPTransBlock, not GraphAttention
                drop_path_rate=self.drop_path_rate,
                irreps_mlp_mid=self.irreps_mlp_mid,
                norm_layer=self.norm_layer,
                tp_path_norm=self.outhead_tp_path_norm,
                tp_irrep_norm=self.outhead_tp_irrep_norm,
                affine_ln=self.affine_ln,
                # only GraphAttention
                normalization=self.outhead_tp_irrep_norm,
                path_normalization=self.outhead_tp_path_norm,
            )

        self.apply(self._init_weights)

    def build_blocks(self):
        """N blocks of: Layer Norm 1 -> DotProductAttention -> Layer Norm 2 -> FeedForwardNetwork
        Last block outputs scalars (l0) only.
        """
        for i in range(self.num_layers):
            # last layer is different which will screw up DEQ
            # last block outputs scalars only (l0 only, no higher l's)
            # irreps_node_embedding -> irreps_feature
            # "128x0e+64x1e+32x2e" -> "512x0e"
            if i != (self.num_layers - 1):
                # input == output == [num_atoms, 512]
                irreps_block_output = self.irreps_node_embedding
            else:
                # [num_atoms, 480]
                irreps_block_output = self.irreps_feature
            # Layer Norm 1 -> DotProductAttention -> Layer Norm 2 -> FeedForwardNetwork
            # extra 'input injection' (= everything except node_features) used in DotProductAttention
            blk = DPTransBlock(
                irreps_node_input=self.irreps_node_embedding,
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

    # the gradient of energy is following the implementation here:
    # https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/models/spinconv.py#L186
    @torch.enable_grad()
    def forward(self, node_atom, pos, batch, **kwargs) -> torch.Tensor:

        # pos = node feature matrix
        if self.force_block is None:
            pos = pos.requires_grad_(True)

        # get graph edges based on radius
        edge_index = radius_graph(
            x=pos, r=self.max_radius, batch=batch, max_num_neighbors=1000
        )

        # encode edges
        edge_src, edge_dst = edge_index[0], edge_index[1]
        edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
        # radial basis function embedding of edge length
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = self.rbf(edge_length)
        # spherical harmonics embedding of edge vector
        edge_sh = o3.spherical_harmonics(
            l=self.irreps_edge_attr,
            x=edge_vec,
            normalize=True,
            normalization="component",
        )

        # encode atom type z_i
        atom_embedding, atom_attr, atom_onehot = self.atom_embed(node_atom)

        # Constant One, r_ij -> Linear, Depthwise TP, Linear, Scaled Scatter
        edge_degree_embedding = self.edge_deg_embed(
            # atom_embedding is just used for the shape
            atom_embedding,
            edge_sh,
            edge_length_embedding,
            edge_src,
            edge_dst,
            batch,
        )

        # node_features = x
        node_features = atom_embedding + edge_degree_embedding

        # node_attr = ?
        # node_attr = torch.ones_like(node_features.narrow(dim=1, start=0, length=1))
        node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

        ###############################################################
        # Update node embeddings
        ###############################################################

        for blknum, blk in enumerate(self.blocks):
            node_features = blk(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,
                edge_scalars=edge_length_embedding,
                # drop_path = GraphDropPath(drop_path_rate) uses batch
                batch=batch,
            )

        node_features = self.final_block(
            node_input=node_features,
            node_attr=node_attr,
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_attr=edge_sh,
            edge_scalars=edge_length_embedding,
            batch=batch,
        )
        # print(f'After final block: {node_features.shape}')

        # Layer Norm
        node_features = self.norm(node_features, batch=batch)
        if self.out_dropout is not None:
            node_features = self.out_dropout(node_features)

        # output head
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

        ###############################################################
        # Force estimation
        ###############################################################

        if self.force_block is None:
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
                )[0]
            )
        else:
            # atom-wise forces using a block of equivariant graph attention
            # and treating the output of degree 1 as the predictions
            # forces = self.force_block(x, atomic_numbers, edge_distance, edge_index)
            # x: [num_atoms*batch_size, irrep_dim]
            # forces: o3.Irreps("1x1e") -> [num_atoms*batch_size, 3]
            forces = self.force_block(
                node_input=node_features,
                node_attr=node_attr,
                edge_src=edge_src,
                edge_dst=edge_dst,
                edge_attr=edge_sh,  # requires_grad
                edge_scalars=edge_length_embedding,  # requires_grad
                batch=batch,
            )
            # in case we predict more irreps than forces, select the first 3
            # forces = forces.embedding.narrow(1, 1, 3)
            if forces.shape[1] != 3:
                # [num_atoms*batch_size, 3, 1]
                forces = forces.narrow(1, 1, 3)
            if len(forces.shape) > 2:
                # [num_atoms*batch_size, 3]
                forces = forces.view(-1, 3)

        return energy, forces, {}


@register_model
def dot_product_attention_transformer_exp_l2_md17(
    irreps_in,
    max_radius,
    number_of_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    num_layers=6,
    irreps_node_attr="1x0e",
    irreps_node_embedding="128x0e+64x1e+32x2e",
    irreps_sh="1x0e+1x1e+1x2e",
    fc_neurons=[64, 64],
    basis_type="exp",
    irreps_feature="512x0e",
    irreps_head="32x0e+16x1e+8x2e",
    num_heads=4,
    irreps_pre_attn=None,
    rescale_degree=False,
    nonlinear_message=False,
    irreps_mlp_mid="384x0e+192x1e+96x2e",
    norm_layer="layer",
    alpha_drop=0.0,
    proj_drop=0.0,
    out_drop=0.0,
    drop_path_rate=0.0,
    scale=None,
    **kwargs,
):
    model = DotProductAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding=irreps_node_embedding,
        num_layers=num_layers,
        irreps_node_attr=irreps_node_attr,
        irreps_sh=irreps_sh,
        max_radius=max_radius,
        number_of_basis=number_of_basis,
        fc_neurons=fc_neurons,
        basis_type=basis_type,
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
        task_mean=task_mean,
        task_std=task_std,
        scale=scale,
        atomref=atomref,
        **kwargs,
    )
    return model


@register_model
def dot_product_attention_transformer_exp_l3_md17(
    irreps_in,
    max_radius,
    number_of_basis=128,
    atomref=None,
    task_mean=None,
    task_std=None,
    irreps_node_embedding="128x0e+64x1e+64x2e+32x3e",
    num_layers=6,
    irreps_node_attr="1x0e",
    irreps_sh="1x0e+1x1e+1x2e+1x3e",
    fc_neurons=[64, 64],
    basis_type="exp",
    irreps_feature="512x0e",
    irreps_head="32x0e+16x1e+16x2e+8x3e",
    num_heads=4,
    irreps_pre_attn=None,
    rescale_degree=False,
    nonlinear_message=False,
    irreps_mlp_mid="384x0e+192x1e+192x2e+96x3e",
    norm_layer="layer",
    alpha_drop=0.0,
    proj_drop=0.0,
    out_drop=0.0,
    drop_path_rate=0.0,
    scale=None,
    **kwargs,
):
    model = DotProductAttentionTransformerMD17(
        irreps_in=irreps_in,
        irreps_node_embedding=irreps_node_embedding,
        num_layers=6,
        irreps_node_attr=irreps_node_attr,
        irreps_sh=irreps_sh,
        max_radius=max_radius,
        number_of_basis=number_of_basis,
        fc_neurons=fc_neurons,
        basis_type=basis_type,
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
        task_mean=task_mean,
        task_std=task_std,
        scale=scale,
        atomref=atomref,
        **kwargs,
    )
    return model
