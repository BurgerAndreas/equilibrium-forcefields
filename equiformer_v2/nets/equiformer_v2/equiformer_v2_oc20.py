import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pyexpat.model import XML_CQUANT_OPT

import wandb
import copy

from ocpmodels.common.registry import registry
from ocpmodels.common.utils import conditional_grad
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.sampling import CalcSpherePoints
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    LinearSigmoidSmearing,
    SigmoidSmearing,
    SiLUSmearing,
)

try:
    from e3nn import o3
except ImportError:
    pass

from .gaussian_rbf import GaussianRadialBasisLayer
from torch.nn import Linear
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2,
)
from .module_list import ModuleListInfo
from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    EmbFeedForwardNetwork,
    TransBlockV2,
)
from .input_block import EdgeDegreeEmbedding

import deq2ff.logging_utils_deq as logging_utils_deq

# Statistics of OC20 IS2RE 100K
# IS2RE: 100k, max_radius = 5, max_neighbors = 100
# _AVG_NUM_NODES = 77.81317
# _AVG_DEGREE = 23.395238876342773


@registry.register_model("equiformer_v2_oc20")
class EquiformerV2_OC20(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        ln_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        path_drop (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """

    def __init__(
        self,
        # not used but necessary for OC20 compatibility
        num_atoms=None,  # not used
        bond_feat_dim=None,  # not used
        num_targets=None,  # not used
        use_pbc=True,
        regress_forces=True,
        forces_via_grad=False,
        otf_graph=True,
        max_neighbors=500,
        max_radius=5.0,
        max_num_elements=90,
        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        ln_type="rms_norm_sh",
        ln_norm="component",  # component, norm
        ln_affine=True,
        # num_coefficients = sum_i int((lmax_list[i] + 1) ** 2)
        # lmax_list=[3] -> num_coefficients = 16
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512,
        attn_activation="scaled_silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        # dropout rates
        alpha_drop=0.1,
        path_drop=0.05,
        proj_drop=0.0,
        head_alpha_drop=0.0,  # added and decreases accuracy
        weight_init="normal",
        # added
        use_variational_alpha_drop=False,
        use_variational_path_drop=False,
        # Statistics of IS2RE 100K
        # IS2RE: 100k, max_radius = 5, max_neighbors = 100
        _AVG_NUM_NODES=77.81317,
        _AVG_DEGREE=23.395238876342773,
        task_mean=None,
        task_std=None,
        name=None,
        force_head="SO2EquivariantGraphAttention",
        energy_head="FeedForwardNetwork",
        force_scale_head=None,
        fsbv=1,  # TODO: temporary
        skip_blocks=False,
        batchify_for_torchdeq=False,
        edge_emb_st_max_norm=None,
        ln="pre",
        enc_ln=False,
        final_ln=False,
        **kwargs,
    ):
        super().__init__()

        if len(kwargs) > 0:
            print(
                f"-"*50 + "\n"
                f"Ignoring kwargs in {self.__class__.__name__}:", kwargs
            )

        # added
        self._AVG_NUM_NODES = _AVG_NUM_NODES
        self._AVG_DEGREE = _AVG_DEGREE
        self.task_mean = task_mean
        self.task_std = task_std

        self.energy_head = energy_head
        self.force_head = force_head
        self.force_scale_head = force_scale_head

        self.skip_blocks = skip_blocks

        self.batchify_for_torchdeq = batchify_for_torchdeq

        # print(
        #     "Number of trainable params:",
        #     sum(p.numel() for p in self.parameters() if p.requires_grad),
        # )

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        # compute forces via gradient of the energy w.r.t. positions
        self.forces_via_grad = forces_via_grad
        # compute forces directly via prediction head
        self.direct_forces = (
            regress_forces and not forces_via_grad
        )  # for ocp models conditional_grad
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.ln_type = ln_type
        self.ln_norm = ln_norm
        self.ln_affine = ln_affine

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.path_drop = path_drop
        self.proj_drop = proj_drop
        self.head_alpha_drop = head_alpha_drop

        self.use_variational_alpha_drop = use_variational_alpha_drop
        self.use_variational_path_drop = use_variational_path_drop

        self.weight_init = weight_init
        assert self.weight_init in [
            "normal",
            "uniform",
        ], f"Unknown weight_init: {self.weight_init}"

        self.device = "cpu"  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
        ]
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
            # self.distance_expansion = GaussianRadialBasisLayer(number_of_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(
            "({}, {})".format(max(self.lmax_list), max(self.lmax_list))
        )
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=self.grid_resolution, normalization="component"
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self._AVG_DEGREE,
            st_max_norm=edge_emb_st_max_norm,
        )

        self.ln = ln
        self.final_ln = final_ln

        self.build_blocks()

        if enc_ln:
            self.norm_enc = get_normalization_layer(
                self.ln_type,
                lmax=max(self.lmax_list),
                num_channels=self.sphere_channels,
            )
        else:
            self.norm_enc = None

        # Output blocks for energy and forces
        # normalization before output blocks
        self.norm = get_normalization_layer(
            self.ln_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels
        )
        # FeedForwardNetwork, SO2EquivariantGraphAttention
        if self.energy_head == "FeedForwardNetwork":
            self.energy_block = FeedForwardNetwork(
                sphere_channels=self.sphere_channels,
                hidden_channels=self.ffn_hidden_channels,
                output_channels=1,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_grid=self.SO3_grid,
                activation=self.ffn_activation,
                use_gate_act=self.use_gate_act,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
            )
        elif self.energy_head == "SO2EquivariantGraphAttention":
            self.energy_block = SO2EquivariantGraphAttention(
                sphere_channels=self.sphere_channels,
                hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                output_channels=1,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.block_use_atom_edge_embedding,
                use_m_share_rad=self.use_m_share_rad,
                activation=self.attn_activation,
                use_s2_act_attn=self.use_s2_act_attn,
                use_attn_renorm=self.use_attn_renorm,
                use_gate_act=self.use_gate_act,
                use_sep_s2_act=self.use_sep_s2_act,
                # dropout
                alpha_drop=self.head_alpha_drop,
            )
        else:
            raise ValueError(
                f"Unknown energy_head: {self.energy_head}. Try model.energy_head=SO2EquivariantGraphAttention"
            )
        if self.regress_forces:
            # self.force_block = SO2EquivariantGraphAttention(
            self.force_block = eval(force_head)(
                sphere_channels=self.sphere_channels,
                hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                output_channels=1,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.block_use_atom_edge_embedding,
                use_m_share_rad=self.use_m_share_rad,
                activation=self.attn_activation,  # unused
                use_s2_act_attn=self.use_s2_act_attn,
                use_attn_renorm=self.use_attn_renorm,
                use_gate_act=self.use_gate_act,
                use_sep_s2_act=self.use_sep_s2_act,
                # dropout
                alpha_drop=self.head_alpha_drop,
            )
        if self.force_scale_head == "FeedForwardNetwork":
            self.force_scale_block = FeedForwardNetwork(
                sphere_channels=self.sphere_channels,
                hidden_channels=self.ffn_hidden_channels,
                output_channels=1,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_grid=self.SO3_grid,
                activation=self.ffn_activation,
                use_gate_act=self.use_gate_act,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
            )
        elif self.force_scale_head == "EmbFeedForwardNetwork":
            self.force_scale_block = EmbFeedForwardNetwork(
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                # FeedForwardNetwork
                sphere_channels=self.sphere_channels,
                hidden_channels=self.ffn_hidden_channels,
                output_channels=1,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_grid=self.SO3_grid,
                activation=self.ffn_activation,
                use_gate_act=self.use_gate_act,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                version=fsbv,
            )
        elif self.force_scale_head == "SO2EquivariantGraphAttention":
            self.force_scale_block = SO2EquivariantGraphAttention(
                sphere_channels=self.sphere_channels,
                hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                output_channels=1,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.block_use_atom_edge_embedding,
                use_m_share_rad=self.use_m_share_rad,
                activation=self.attn_activation,
                use_s2_act_attn=self.use_s2_act_attn,
                use_attn_renorm=self.use_attn_renorm,
                use_gate_act=self.use_gate_act,
                use_sep_s2_act=self.use_sep_s2_act,
                # dropout
                alpha_drop=self.head_alpha_drop,
            )
        elif self.force_scale_head in [False, None, "None"]:
            self.force_scale_block = None
        else:
            raise ValueError(
                f"Unknown force_scale_head: {self.force_scale_head}. Try model.force_scale_head=FeedForwardNetwork"
            )

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def build_blocks(self):
        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            sphere_channels_in = self.sphere_channels
            block = TransBlockV2(
                # sphere_channels=self.sphere_channels,
                sphere_channels=sphere_channels_in,
                attn_hidden_channels=self.attn_hidden_channels,
                num_heads=self.num_heads,
                attn_alpha_channels=self.attn_alpha_channels,
                attn_value_channels=self.attn_value_channels,
                ffn_hidden_channels=self.ffn_hidden_channels,
                output_channels=self.sphere_channels,
                lmax_list=self.lmax_list,
                mmax_list=self.mmax_list,
                SO3_rotation=self.SO3_rotation,
                mappingReduced=self.mappingReduced,
                SO3_grid=self.SO3_grid,
                max_num_elements=self.max_num_elements,
                edge_channels_list=self.edge_channels_list,
                use_atom_edge_embedding=self.block_use_atom_edge_embedding,
                use_m_share_rad=self.use_m_share_rad,
                attn_activation=self.attn_activation,  # unused
                use_s2_act_attn=self.use_s2_act_attn,
                use_attn_renorm=self.use_attn_renorm,
                ffn_activation=self.ffn_activation,
                use_gate_act=self.use_gate_act,
                use_grid_mlp=self.use_grid_mlp,
                use_sep_s2_act=self.use_sep_s2_act,
                ln_type=self.ln_type,
                # dropout
                alpha_drop=self.alpha_drop,
                path_drop=self.path_drop,
                proj_drop=self.proj_drop,
                # added
                use_variational_alpha_drop=self.use_variational_alpha_drop,
                use_variational_path_drop=self.use_variational_path_drop,
                ln_norm=self.ln_norm,
                ln_affine=self.ln_affine,
                ln=self.ln,
                final_ln=self.final_ln,
            )
            self.blocks.append(block)

    def get_shapes(self, data, **kwargs):
        """Return dictionary of shapes."""

        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        if hasattr(data, "atomic_numbers"):
            atomic_numbers = data.atomic_numbers.long()
        else:
            # MD17
            atomic_numbers = data.z.long()
            data.atomic_numbers = data.z

        # When using MD17 instead of OC20
        # cell is not used unless (otf_graph is False) or (use_pbc is not None)
        if not hasattr(data, "cell"):
            data.cell = None

        # molecules in batch can be of different sizes
        num_atoms = len(atomic_numbers)
        pos = data.pos

        # basically the same as edge_src, edge_dst, edge_vec, edge_length in V1
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data=data, pos=pos)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        # data unused
        edge_rot_mat = self._init_edge_rot_mat(
            edge_index=edge_index, edge_distance_vec=edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        # shape: [num_atoms*batch_size, num_coefficients, num_channels]
        x: SO3_Embedding = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )
        # both: [num_atoms, num_coefficients, num_channels]
        # num_coefficients = sum([(2 * l + 1) for l in self.lmax_list])
        # addition, not concatenation
        x.embedding = x.embedding + edge_degree.embedding

        edge_src = edge_index[0]
        logs = {
            "NumNodes": x.embedding.shape[0],  # num_atoms * batch_size
            "NumEdges": edge_src.shape[0],
            "DimInputInjection": x.embedding.shape[1],
            "DimFixedPoint": x.embedding.shape[1],
            "NodeEmbeddingShape": x.embedding.shape,
        }
        return logs

    def measure_oversmoothing(self, x, batch, step=None, split=None, layer=None):
        """
        From https://arxiv.org/pdf/1909.03211:
        Mean Average Distance (MAD), which calculates the mean average distance
        among node representations in the graph to measure the smoothness of the graph
        (smoothness means similarity of graph nodes representation)
        x: node representations
        """
        similarity_euclidean = []
        similarity_cosine = []
        for b in range(max(batch) + 1):
            mask = batch == b  # [NB]
            if mask.sum() == 0:
                # empty batch
                continue
            # x[mask]: [N, D, C]
            # x[n]: [D, C]
            idx = torch.arange(x.shape[0], device=x.device)[mask]
            for i, n1 in enumerate(idx):
                for n2 in idx[i + 1 :]:
                    similarity_euclidean.append(torch.norm(x[n1] - x[n2], p=2).detach())
                    # dim=0 or dim=1?
                    similarity_cosine.append(
                        F.cosine_similarity(x[n1], x[n2]).mean().detach()
                    )
            # one batch is enough
            break
        _logs = {
            "sim_eucl_mean": torch.mean(torch.tensor(similarity_euclidean)),
            "sim_cos_mean": torch.mean(torch.tensor(similarity_cosine)),
            # "sim_eucl_max": torch.max(torch.tensor(similarity_euclidean)),
            # "sim_cos_max": torch.max(torch.tensor(similarity_cosine)),
            # "sim_eucl_min": torch.min(torch.tensor(similarity_euclidean)),
            # "sim_cos_min": torch.min(torch.tensor(similarity_cosine)),
        }
        if split is not None:
            _logs = {k + f"_{split}": v for k, v in _logs.items()}
        if layer is not None:
            _logs = {k + f"_l{layer}": v for k, v in _logs.items()}
        if step is not None:
            wandb.log(_logs, step=step)
        return _logs

    def reset_dropout(self, x, batch):
        # set dropout mask
        for i in range(self.num_layers):
            # torch.nn.Dropout won't have .update_mask
            if callable(
                getattr(
                    self.blocks[i].graph_attention.alpha_dropout, "update_mask", None
                )
            ):
                # shape will vary with each batch, since it depends on the number of edges
                # which depends on the molecule configuration
                # mask_before = self.blocks[i].graph_attention.alpha_dropout.mask
                alpha_mask = self.blocks[i].graph_attention.alpha_dropout.update_mask(
                    shape=[self.num_edges, 1, self.num_heads, 1],  # TODO: check shape
                    dtype=x.dtype,
                    device=x.device,
                )
            if self.blocks[i].path_drop is not None:
                # mask_before = self.blocks[i].path_drop.mask
                path_mask = self.blocks[i].path_drop.update_mask(x=x, batch=batch)
            if self.blocks[i].proj_drop is not None:
                self.blocks[i].proj_drop.update_mask(x=x, batch=batch)
            # adding noise to hidden states
            if self.blocks[i].noise_in is not None:
                self.blocks[i].noise_in.update_mask(x.shape, x.dtype, x.device)
            if self.blocks[i].noise_out is not None:
                self.blocks[i].noise_out.update_mask(x.shape, x.dtype, x.device)

    def set_current_deq(self, reuse=False):
        """We use different DEQ solvers for training, evaluation, and fixed-point reuse."""
        pass

    @conditional_grad(torch.enable_grad())
    def encode(self, data):
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        if hasattr(data, "atomic_numbers"):
            atomic_numbers = data.atomic_numbers.long()
        else:
            # MD17
            atomic_numbers = data.z.long()
            data.atomic_numbers = data.z

        # When using MD17 instead of OC20
        # cell is not used unless (otf_graph is False) or (use_pbc is not None)
        if not hasattr(data, "cell"):
            data.cell = None

        num_atoms = len(atomic_numbers)
        self.num_atoms = num_atoms

        # pos = data.pos.clone()
        # pos = data.pos.detach()
        pos = copy.deepcopy(data.pos.detach())
        if self.forces_via_grad:
            # data.pos.requires_grad_(True)
            pos.requires_grad_(True)

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self.generate_graph(data=data, pos=pos)

        self.num_edges = edge_distance.shape[0]

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(
            edge_index=edge_index, edge_distance_vec=edge_distance_vec
        )

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )
        self.shape_batched = x.embedding.shape
        self.shape_unbatched = (self.batch_size, num_atoms, *x.embedding.shape[1:])

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        # radial basis function
        # E1: edge_length_embedding = self.rbf(edge_length)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )
        x.embedding = x.embedding + edge_degree.embedding

        if self.norm_enc is not None:
            x.embedding = self.norm_enc(x.embedding)

        # logging
        # if step is not None:
        #     # log the input injection (output of encoder)
        #     logging_utils_deq.log_fixed_point_norm(
        #         x.embedding.clone().detach(), step, datasplit, name="emb"
        #     )

        return x, pos, atomic_numbers, edge_distance, edge_index

    # from OCP models to predict F=dE/dx
    # not needed since we are predicting forces directly with another head
    # @torch.compile
    @conditional_grad(torch.enable_grad())
    def forward(
        self, data, step=None, datasplit=None, return_fixedpoint=False, fixedpoint=None, **kwargs
    ):

        x, pos, atomic_numbers, edge_distance, edge_index = self.encode(data)

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        self.reset_dropout(x.embedding, data.batch)

        # if self.skip_blocks:
        #     pass
        # else:
        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,  # for GraphPathDrop
            )
            # if step is not None and (step % 100 == 0) or datasplit in ["val", "test"]:
            #     self.measure_oversmoothing(x=x.embedding, batch=data.batch, step=step, split=datasplit, layer=i)

        # if step is not None and (step % 100 == 0) or datasplit in ["val", "test"]:
        #     self.measure_oversmoothing(x=x.embedding, batch=data.batch, step=step, split=datasplit)

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        # corresponding to DEQ
        info = {
            "nstep": torch.tensor(
                [self.num_layers] * x.embedding.shape[0],
                dtype=torch.float16,
                device=x.embedding.device,
            )
        }

        ######################################################
        # Logging
        ######################################################
        # if step is not None:
        #     logging_utils_deq.log_fixed_point_norm(
        #         x.embedding.clone().detach(), step, datasplit
        #     )

        return self.decode(
            data=data,
            x=x,
            fp=x.embedding.detach(),  # last fixed-point estimate
            pos=pos,
            atomic_numbers=atomic_numbers,
            edge_distance=edge_distance,
            edge_index=edge_index,
            info=info,
            return_fixedpoint=return_fixedpoint,
        )

    @conditional_grad(torch.enable_grad())
    def decode(
        self,
        data,
        x,
        fp,
        pos,
        atomic_numbers,
        edge_distance,
        edge_index,
        info,
        return_fixedpoint=False,
    ):
        """Predict energy and forces from fixed-point estimate.
        Uses separate heads for energy and forces.
        """

        ######################################################
        # Logging
        ######################################################
        # if step is not None:
        #     # log the final fixed-point
        #     logging_utils_deq.log_fixed_point_norm(z_pred, step, datasplit)
        #     # log the input injection (output of encoder)
        #     logging_utils_deq.log_fixed_point_norm(emb, step, datasplit, name="emb")

        ###############################################################
        # Energy estimation
        ###############################################################
        # (B, num_coefficients, 1)
        node_energy = self.energy_block(x)
        # (B, 1, 1)
        node_energy = node_energy.embedding.narrow(dim=1, start=0, length=1)
        energy = torch.zeros(
            len(data.natoms), device=node_energy.device, dtype=node_energy.dtype
        )
        # basically predict an energy for each atom and sum them up
        # node_energy.view(-1): [num_atoms*batch_size]
        # data.batch: [num_atoms*batch_size]
        # data.batch contains the batch index for each atom (node)
        # [0, ..., 0, 1, ..., 1, ..., B-1, ..., B-1]
        energy.index_add_(dim=0, index=data.batch, source=node_energy.view(-1))
        energy = energy / self._AVG_NUM_NODES

        ###############################################################
        # Force estimation
        ###############################################################
        if self.regress_forces:
            if self.forces_via_grad:
                # [num_atoms*batch_size, 3]
                forces = -1 * (
                    torch.autograd.grad(
                        energy,
                        pos,
                        grad_outputs=torch.ones_like(energy),
                        # we need to retain the graph to call loss.backward() later
                        create_graph=True,
                        # retain_graph=True,
                    )[0]
                )

            else:
                # atom-wise forces using a block of equivariant graph attention
                # and treating the output of degree 1 as the predictions
                # x: [num_atoms*batch_size, num_coefficients, sphere_channels]
                # forces: [num_atoms*batch_size, num_coefficients, 1]
                forces = self.force_block(x, atomic_numbers, edge_distance, edge_index)
                # [num_atoms*batch_size, 3, 1]
                forces = forces.embedding.narrow(dim=1, start=1, length=3)
                # [num_atoms*batch_size, 3]
                forces = forces.view(-1, 3)
                # multiply force on each node by a scalar
                if self.force_scale_block is not None:
                    if self.force_scale_head == "FeedForwardNetwork":
                        force_scale = self.force_scale_block(x)
                    else:  # SO2EquivariantGraphAttention
                        force_scale = self.force_scale_block(
                            x, atomic_numbers, edge_distance, edge_index
                        )
                    # select scalars only, one per node # (B, 1, 1)
                    force_scale = force_scale.embedding.narrow(dim=1, start=0, length=1)
                    # view: [B, 1]
                    force_scale = force_scale.view(-1, 1)
                    # [B, 3]
                    force_scale = force_scale.expand(-1, 3)
                    forces = forces * force_scale

        # info = {} # Todo@temp for debugging
        if self.regress_forces:
            if return_fixedpoint:
                # z_pred = sampled fixed point trajectory (tracked gradients)
                return energy, forces, fp, info
            return energy, forces, info
        else:
            if return_fixedpoint:
                # z_pred = sampled fixed point trajectory (tracked gradients)
                return energy, fp, info
            return energy, info

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, edge_index, edge_distance_vec, data=None):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            # if m.affine:
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        """?"""
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) or isinstance(
                        module, SO3_LinearV2
                    ):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)
