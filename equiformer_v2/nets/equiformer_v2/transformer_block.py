import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch_geometric
import copy

from .activation import (
    ScaledSiLU,
    ScaledSwiGLU,
    SwiGLU,
    ScaledSmoothLeakyReLU,
    SmoothLeakyReLU,
    GateActivation,
    SeparableS2Activation,
    S2Activation,
    activations_fn,
)
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    get_normalization_layer,
)
from .so2_ops import SO2_Convolution, SO2_Linear
from .so3 import SO3_Embedding, SO3_Linear, SO3_LinearV2
from .radial_function import RadialFunction
from .drop import (
    GraphPathDrop,
    VariationalGraphPathDrop,
    VariationalDropout,
    EquivariantDropoutArraySphericalHarmonics,
    RecurrentNoise,
)

from deq2ff.logging_utils_deq import print_values


class SO2EquivariantGraphAttention(torch.nn.Module):
    """
    SO2EquivariantGraphAttention: Perform MLP attention + non-linear message passing
        SO(2) Convolution with radial function -> S2 Activation -> SO(2) Convolution -> attention weights and non-linear messages
        attention weights * non-linear messages -> Linear

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during the SO(2) conv
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        output_channels (int):      Number of output channels
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        activation (str):           Type of activation function
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        output_channels,
        lmax_list,
        mmax_list,
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        use_m_share_rad=False,
        activation="scaled_silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        use_gate_act=False,
        use_sep_s2_act=True,
        alpha_drop=0.0,
        # added
        use_variational_alpha_drop=False,
        **kwargs,
    ):
        super(SO2EquivariantGraphAttention, self).__init__()

        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(self.lmax_list)

        self.SO3_rotation = SO3_rotation
        self.mappingReduced = mappingReduced
        self.SO3_grid = SO3_grid

        # Create edge scalar (invariant to rotations) features
        # Embedding function of the atomic numbers
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad

        if self.use_atom_edge_embedding:
            # lookup table for atomic numbers
            self.source_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
            nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.use_gate_act = use_gate_act
        self.use_sep_s2_act = use_sep_s2_act

        assert not self.use_s2_act_attn  # since this is not used

        # Create SO(2) convolution blocks
        extra_m0_output_channels = None
        if not self.use_s2_act_attn:  # False by default
            extra_m0_output_channels = self.num_heads * self.attn_alpha_channels
            if self.use_gate_act:  # False by default
                extra_m0_output_channels = (
                    extra_m0_output_channels
                    + max(self.lmax_list) * self.hidden_channels
                )
            else:
                if self.use_sep_s2_act:
                    extra_m0_output_channels = (
                        extra_m0_output_channels + self.hidden_channels
                    )

        if self.use_m_share_rad:
            self.edge_channels_list = self.edge_channels_list + [
                2 * self.sphere_channels * (max(self.lmax_list) + 1)
            ]
            self.rad_func = RadialFunction(
                self.edge_channels_list, activation=activation
            )
            expand_index = torch.zeros([(max(self.lmax_list) + 1) ** 2]).long()
            for l in range(max(self.lmax_list) + 1):
                start_idx = l**2
                length = 2 * l + 1
                expand_index[start_idx : (start_idx + length)] = l
            self.register_buffer("expand_index", expand_index)

        self.so2_conv_1 = SO2_Convolution(
            sphere_channels=2 * self.sphere_channels,
            m_output_channels=self.hidden_channels,
            lmax_list=self.lmax_list,
            mmax_list=self.mmax_list,
            mappingReduced=self.mappingReduced,
            internal_weights=(False if not self.use_m_share_rad else True),
            edge_channels_list=(
                self.edge_channels_list if not self.use_m_share_rad else None
            ),
            extra_m0_output_channels=extra_m0_output_channels,  # for attention weights and/or gate activation
        )

        if self.use_s2_act_attn:  # False by default
            self.alpha_norm = None
            self.alpha_act = None
            self.alpha_dot = None
        else:
            if self.use_attn_renorm:  # True by default
                self.alpha_norm = torch.nn.LayerNorm(self.attn_alpha_channels)
            else:
                self.alpha_norm = torch.nn.Identity()
            # activation
            self.alpha_act = SmoothLeakyReLU()
            self.alpha_dot = torch.nn.Parameter(
                torch.randn(self.num_heads, self.attn_alpha_channels)
            )
            # torch_geometric.nn.inits.glorot(self.alpha_dot) # Following GATv2
            std = 1.0 / math.sqrt(self.attn_alpha_channels)
            torch.nn.init.uniform_(self.alpha_dot, -std, std)

        self.alpha_dropout = None
        if alpha_drop != 0.0:
            if use_variational_alpha_drop:
                # print(f"{self.__class__.__name__}: Using VariationalDropout (Alpha).")
                self.alpha_dropout = VariationalDropout(alpha_drop)
            else:
                # print(
                #     f"{self.__class__.__name__}: Not using VariationalDropout (Alpha)."
                # )
                self.alpha_dropout = torch.nn.Dropout(alpha_drop)

        if self.use_gate_act:  # False by default
            self.graph_attentionte_act = GateActivation(
                lmax=max(self.lmax_list),
                mmax=max(self.mmax_list),
                num_channels=self.hidden_channels,
                # scalar_activation = 'silu',
                # gate_activation = 'sigmoid',
            )
        else:
            if self.use_sep_s2_act:
                # separable S2 activation
                self.s2_act = SeparableS2Activation(
                    lmax=max(self.lmax_list),
                    mmax=max(self.mmax_list),
                    # added
                    activation=activation,
                )
            else:
                # S2 activation
                self.s2_act = S2Activation(
                    lmax=max(self.lmax_list),
                    mmax=max(self.mmax_list),
                    # added
                    activation=activation,
                )

        self.so2_conv_2 = SO2_Convolution(
            self.hidden_channels,
            self.num_heads * self.attn_value_channels,
            self.lmax_list,
            self.mmax_list,
            self.mappingReduced,
            internal_weights=True,
            edge_channels_list=None,
            extra_m0_output_channels=(
                self.num_heads if self.use_s2_act_attn else None
            ),  # for attention weights
        )

        self.proj = SO3_LinearV2(
            in_features=self.num_heads * self.attn_value_channels,
            out_features=self.output_channels,
            lmax=self.lmax_list[0],
        )

    def forward(self, x, atomic_numbers, edge_distance, edge_index, **kwargs):
        """atom_edge_embedding, S03_Embedding, SO(2)-convolution, Activation, SO(2)-convolution, Attention, Projection (SO3_LinearV2)"""
        # x: [B, D, C]
        # atomic_numbers: [B]
        # edge_index: [2, E]
        # edge_distance: [E, edim]

        # Compute edge scalar features (invariant to rotations)
        # Uses atomic numbers and edge distance as inputs
        if self.use_atom_edge_embedding:  # True by default
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            # [E, C]
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )  # [E, edim+2*C]
        else:
            x_edge = edge_distance  # [E, edim]

        x_source = x.clone()
        x_target = x.clone()
        # [B, D, C] -> [E, D, C]
        x_source._expand_edge(edge_index[0, :])
        x_target._expand_edge(edge_index[1, :])

        # [B, D, C*2]
        x_message_data = torch.cat((x_source.embedding, x_target.embedding), dim=2)
        x_message = SO3_Embedding(
            0,
            x_target.lmax_list.copy(),
            x_target.num_channels * 2,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        x_message.set_embedding(x_message_data)
        x_message.set_lmax_mmax(self.lmax_list.copy(), self.mmax_list.copy())

        # radial function (scale all m components within a type-L vector of one channel with the same weight)
        if self.use_m_share_rad:  # False by default
            x_edge_weight = self.rad_func(x_edge)
            x_edge_weight = x_edge_weight.reshape(
                -1, (max(self.lmax_list) + 1), 2 * self.sphere_channels
            )
            x_edge_weight = torch.index_select(
                x_edge_weight, dim=1, index=self.expand_index
            )  # [E, (L_max + 1) ** 2, C]
            x_message.embedding = x_message.embedding * x_edge_weight

        # Rotate the irreps to align with the edge
        # [B, D, C*2] -> [B, D', C*2]
        x_message._rotate(self.SO3_rotation, self.lmax_list, self.mmax_list)

        # First SO(2)-convolution
        if self.use_s2_act_attn:  # False by default
            x_message = self.so2_conv_1(x_message, x_edge)
        else:
            # x_message: [E, D', A] # A = attn_alpha_channels
            # x_0_extra: [E, M] # M = num_heads * attn_alpha_channels + hidden_channels
            x_message, x_0_extra = self.so2_conv_1(x_message, x_edge)

        # Activation
        x_alpha_num_channels = self.num_heads * self.attn_alpha_channels
        if self.use_gate_act:  # False by default
            # Gate activation
            x_0_gating = x_0_extra.narrow(
                1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels
            )  # for activation
            x_0_alpha = x_0_extra.narrow(
                1, 0, x_alpha_num_channels
            )  # for attention weights
            x_message.embedding = self.graph_attentionte_act(
                x_0_gating, x_message.embedding
            )
        else:
            if self.use_sep_s2_act:  # True by default
                # for activation
                x_0_gating = x_0_extra.narrow(
                    1, x_alpha_num_channels, x_0_extra.shape[1] - x_alpha_num_channels
                )  # [E, hidden_channels]
                # for attention weights
                # [E, attn_alpha_channels]
                x_0_alpha = x_0_extra.narrow(1, 0, x_alpha_num_channels)
                # [E, D', attn_alpha_channels]
                x_message.embedding = self.s2_act(
                    x_0_gating, x_message.embedding, self.SO3_grid
                )
            else:
                x_0_alpha = x_0_extra
                x_message.embedding = self.s2_act(x_message.embedding, self.SO3_grid)
            ##x_message._grid_act(self.SO3_grid, self.value_act, self.mappingReduced)

        # Second SO(2)-convolution
        if self.use_s2_act_attn:  # False by default
            x_message, x_0_extra = self.so2_conv_2(x_message, x_edge)
        else:
            # [E, D', C]
            x_message = self.so2_conv_2(x_message, x_edge)

        # Attention weights
        if self.use_s2_act_attn:  # False by default
            alpha = x_0_extra
        else:
            # [E, num_heads, attn_alpha_channels]
            x_0_alpha = x_0_alpha.reshape(-1, self.num_heads, self.attn_alpha_channels)
            x_0_alpha = self.alpha_norm(x_0_alpha)
            x_0_alpha = self.alpha_act(x_0_alpha)
            # [E, num_heads]
            alpha = torch.einsum("bik, ik -> bi", x_0_alpha, self.alpha_dot)
        #
        alpha = torch_geometric.utils.softmax(alpha, edge_index[1])
        # [E, 1, num_heads, 1]
        alpha = alpha.reshape(alpha.shape[0], 1, self.num_heads, 1)
        # alpha dropout
        if self.alpha_dropout is not None:
            alpha = self.alpha_dropout(alpha)

        # Attention weights * non-linear messages
        attn = x_message.embedding
        # [E, D', C] -> [E, D', num_heads, attn_value_channels]
        attn = attn.reshape(
            attn.shape[0], attn.shape[1], self.num_heads, self.attn_value_channels
        )
        attn = attn * alpha
        # [E, D', self.num_heads * self.attn_value_channels]
        attn = attn.reshape(
            attn.shape[0], attn.shape[1], self.num_heads * self.attn_value_channels
        )
        x_message.embedding = attn

        # Rotate back the irreps
        # [E, D', C] -> [E, D, C]
        x_message._rotate_inv(self.SO3_rotation, self.mappingReduced)

        # Compute the sum of the incoming neighboring messages for each target node
        # [B, D, C]
        x_message._reduce_edge(edge_index[1], len(x.embedding))

        # Project
        # [B, D, O] # often O = C
        out_embedding = self.proj(x_message)

        return out_embedding


class FeedForwardNetwork(torch.nn.Module):
    """
    FeedForwardNetwork: Perform feedforward network with S2 activation or gate activation

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax_list,
        mmax_list,
        SO3_grid,
        activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        **kwargs,
    ):
        super(FeedForwardNetwork, self).__init__()
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.max_lmax = max(self.lmax_list)

        self.so3_linear_1 = SO3_LinearV2(
            self.sphere_channels_all, self.hidden_channels, lmax=self.max_lmax
        )
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(
                        self.sphere_channels_all, self.hidden_channels, bias=True
                    ),
                    # nn.SiLU(),
                    activations_fn(activation),
                )
            else:
                self.scalar_mlp = None
            # TODO: activation
            self.grid_mlp = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                # nn.SiLU(),
                activations_fn(activation),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
                # nn.SiLU(),
                activations_fn(activation),
                nn.Linear(self.hidden_channels, self.hidden_channels, bias=False),
            )
        else:
            if self.use_gate_act:
                self.graph_attentionting_linear = torch.nn.Linear(
                    self.sphere_channels_all, self.max_lmax * self.hidden_channels
                )
                self.graph_attentionte_act = GateActivation(
                    self.max_lmax,
                    self.max_lmax,
                    self.hidden_channels,
                    # scalar_activation = 'silu',
                    # gate_activation = 'sigmoid'
                )
            else:
                if self.use_sep_s2_act:
                    self.graph_attentionting_linear = torch.nn.Linear(
                        self.sphere_channels_all, self.hidden_channels
                    )
                    self.s2_act = SeparableS2Activation(
                        self.max_lmax, self.max_lmax, activation=activation
                    )
                else:
                    self.graph_attentionting_linear = None
                    self.s2_act = S2Activation(
                        self.max_lmax, self.max_lmax, activation=activation
                    )
        self.so3_linear_2 = SO3_LinearV2(
            self.hidden_channels, self.output_channels, lmax=self.max_lmax
        )

    def forward(self, x, **kwargs):
        # x: [B, D, C]

        gating_scalars = None
        if self.use_grid_mlp:  # True by default
            if self.use_sep_s2_act:
                # [B, 1, H]
                gating_scalars = self.scalar_mlp(x.embedding.narrow(1, 0, 1))
        else:
            if self.graph_attentionting_linear is not None:
                gating_scalars = self.graph_attentionting_linear(
                    x.embedding.narrow(1, 0, 1)
                )

        # [B, D, C] -> [B, D, H]
        x = self.so3_linear_1(x)

        if self.use_grid_mlp:  # True by default
            # Project to grid
            input_embedding_grid = x.to_grid(self.SO3_grid, lmax=self.max_lmax)
            # Perform point-wise operations
            # [B, G, G, H]
            input_embedding_grid = self.grid_mlp(input_embedding_grid)
            # Project back to spherical harmonic coefficients
            # [B, D, H]
            x._from_grid(input_embedding_grid, self.SO3_grid, lmax=self.max_lmax)

            if self.use_sep_s2_act:  # True by default
                # [B, 1, H]
                x.embedding = torch.cat(
                    (
                        gating_scalars,  # [B, 1, H]
                        # select vectors: [B, D-1, H]
                        x.embedding.narrow(
                            dim=1, start=1, length=x.embedding.shape[1] - 1
                        ),
                    ),
                    dim=1,
                )
        else:
            if self.use_gate_act:
                x.embedding = self.graph_attentionte_act(gating_scalars, x.embedding)
            else:
                if self.use_sep_s2_act:
                    x.embedding = self.s2_act(
                        gating_scalars, x.embedding, self.SO3_grid
                    )
                else:
                    x.embedding = self.s2_act(x.embedding, self.SO3_grid)

        # final linear layer without activation
        # [B, D, H] -> [B, D, O] # often O = C
        x = self.so3_linear_2(x)

        return x


class EmbFeedForwardNetwork(torch.nn.Module):
    """
    FeedForwardNetwork which includes scalar atom_type and edge features.
    Performs feedforward network with S2 activation or gate activation.

    Args:
        sphere_channels (int):      Number of spherical channels
        hidden_channels (int):      Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        activation (str):           Type of activation function
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs.
        use_sep_s2_act (bool):      If `True`, use separable grid MLP when `use_grid_mlp` is True.
    """

    def __init__(
        self,
        # from SO2EquivariantGraphAttention
        max_num_elements,
        edge_channels_list,
        # from FeedForwardNetwork
        sphere_channels,
        hidden_channels,
        output_channels,
        lmax_list,
        mmax_list,
        SO3_grid,
        activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        version=1,
        **kwargs,
    ):
        super().__init__()
        """ from SO2EquivariantGraphAttention """
        # use_atom_edge_embedding = True
        # edge scalar features
        self.max_num_elements = max_num_elements
        self.edge_channels_list = copy.deepcopy(edge_channels_list)
        self.source_embedding = nn.Embedding(
            num_embeddings=self.max_num_elements,
            embedding_dim=self.edge_channels_list[-1],
        )
        self.target_embedding = nn.Embedding(
            num_embeddings=self.max_num_elements,
            embedding_dim=self.edge_channels_list[-1],
        )
        nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
        nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
        self.edge_channels_list[0] = (
            self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        )

        """ My creation """
        self.version = version
        # node scalar features
        self.atom_type_node_embedding = nn.Embedding(
            num_embeddings=self.max_num_elements, embedding_dim=hidden_channels
        )  # [B, C]

        """ from FeedForwardNetwork """
        self.sphere_channels = sphere_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.num_resolutions = len(lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels
        self.SO3_grid = SO3_grid
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.max_lmax = max(self.lmax_list)

        # TODO
        _in = self.sphere_channels_all
        if self.version == 3:
            _in = self.sphere_channels_all + hidden_channels
        self.so3_linear_1 = SO3_LinearV2(
            in_features=_in, out_features=self.hidden_channels, lmax=self.max_lmax
        )
        if self.use_grid_mlp:
            if self.use_sep_s2_act:
                self.scalar_mlp = nn.Sequential(
                    nn.Linear(
                        in_features=self.sphere_channels_all,
                        out_features=self.hidden_channels,
                        bias=True,
                    ),
                    # nn.SiLU(),
                    activations_fn(activation),
                )
            else:
                self.scalar_mlp = None
            # preserves the hidden dim
            self.grid_mlp = nn.Sequential(
                nn.Linear(
                    in_features=self.hidden_channels,
                    out_features=self.hidden_channels,
                    bias=False,
                ),
                # nn.SiLU(),
                activations_fn(activation),
                nn.Linear(
                    in_features=self.hidden_channels,
                    out_features=self.hidden_channels,
                    bias=False,
                ),
                # nn.SiLU(),
                activations_fn(activation),
                nn.Linear(
                    in_features=self.hidden_channels,
                    out_features=self.hidden_channels,
                    bias=False,
                ),
            )
        else:
            if self.use_gate_act:
                self.graph_attentionting_linear = torch.nn.Linear(
                    self.sphere_channels_all, self.max_lmax * self.hidden_channels
                )
                self.graph_attentionte_act = GateActivation(
                    self.max_lmax,
                    self.max_lmax,
                    self.hidden_channels,
                    # scalar_activation = 'silu',
                    # gate_activation = 'sigmoid'
                )
            else:
                if self.use_sep_s2_act:
                    self.graph_attentionting_linear = torch.nn.Linear(
                        self.sphere_channels_all, self.hidden_channels
                    )
                    self.s2_act = SeparableS2Activation(
                        self.max_lmax, self.max_lmax, activation=activation
                    )
                else:
                    self.graph_attentionting_linear = None
                    self.s2_act = S2Activation(
                        self.max_lmax, self.max_lmax, activation=activation
                    )
        # TODO
        hc = self.hidden_channels
        if self.version == 2:
            hc = 2 * self.hidden_channels
        self.so3_linear_2 = SO3_LinearV2(
            in_features=hc, out_features=self.output_channels, lmax=self.max_lmax
        )

    def forward(self, x, atomic_numbers, edge_distance, edge_index, **kwargs):
        # x: [B, D, C]
        # atomic_numbers: [B]
        # edge_index: [2, E]
        # edge_distance: [E, edim]

        if self.version == 1:
            # Compute edge scalar features (invariant to rotations)
            # Uses atomic numbers and edge distance as inputs
            # which atom types sit and the end of the edges. E := num_edges
            source_element = atomic_numbers[edge_index[0]]  # [E]
            target_element = atomic_numbers[edge_index[1]]  # [E]
            source_embedding = self.source_embedding(source_element)  # [E, C]
            target_embedding = self.target_embedding(target_element)  # [E, C]
            # edge_distance: [E, edim]. x_edge: [E, edim+2*C]
            x_edge = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )

            # concat edge scalar features with node embeddings
            # x = SO3_Embedding(
            #     0,
            #     x.lmax_list.copy(),
            #     x.num_channels + x_edge.shape[1],
            #     device=x.device,
            #     dtype=x.dtype,
            # )
        elif self.version == 2:
            x_node = self.atom_type_node_embedding(atomic_numbers)  # [B, H]
        elif self.version == 3:
            x_node = self.atom_type_node_embedding(atomic_numbers)  # [B, H]

        # from FeedForwardNetwork
        # use_grid_mlp = True
        # input_embedding = x
        gating_scalars = None
        if self.use_sep_s2_act:
            # [B, 1, H]
            gating_scalars = self.scalar_mlp(
                x.embedding.narrow(dim=1, start=0, length=1)
            )

        if self.version == 3:
            # [B, H] -> [B, D, H]
            x_node = x_node.unsqueeze(1).expand(-1, x.embedding.shape[1], -1)
            # [B, D, C], [B, D, H] -> [B, D, C+H]
            x_data = torch.cat((x.embedding, x_node), dim=-1)
            x = SO3_Embedding(
                length=0,  # B
                lmax_list=x.lmax_list.copy(),  # D
                num_channels=x.num_channels + x_node.shape[-1],  # C+H
                device=x.device,
                dtype=x.dtype,
            )
            x.set_embedding(x_data)
            # [B, D, C+H] -> [B, D, H]

        # [B, D, C] -> [B, D, H]
        x = self.so3_linear_1(x)

        # Project to grid
        # [B, G, G, H]
        x_grid = x.to_grid(self.SO3_grid, lmax=self.max_lmax)
        # Perform point-wise operations
        # [B, G, G, H]
        x_grid = self.grid_mlp(x_grid)
        # Project back to spherical harmonic coefficients
        # [B, D, H]
        x._from_grid(x_grid, self.SO3_grid, lmax=self.max_lmax)

        if self.use_sep_s2_act:  # True by default
            # [B, D, H]
            x.embedding = torch.cat(
                (
                    gating_scalars,  # [B, 1, H]
                    # select vectors
                    x.embedding.narrow(
                        dim=1, start=1, length=x.embedding.shape[1] - 1
                    ),  # [B, D-1, H]
                ),
                dim=1,
            )

        if self.version == 1:
            pass
            # concat edge scalar features with node embeddings
            # x = SO3_Embedding(
            #     0,
            #     x.lmax_list.copy(),
            #     x.num_channels + x_edge.shape[1],
            #     device=x.device,
            #     dtype=x.dtype,
            # )
            x = self.so3_linear_2(x)
        elif self.version == 2:
            # concat node scalar features with node embeddings
            # [B, H] -> [B, D, H]
            x_node = x_node.unsqueeze(1).expand(-1, x.embedding.shape[1], -1)
            # [B, D, H], [B, D, H] -> [B, D, 2*H]
            x_data = torch.cat((x.embedding, x_node), dim=-1)
            x = SO3_Embedding(
                length=0,  # B
                lmax_list=x.lmax_list.copy(),  # D
                num_channels=x.num_channels + x_node.shape[-1],  # 2*H
                device=x.device,
                dtype=x.dtype,
            )
            x.set_embedding(x_data)
            # [B, D, 2*H] -> [B, D, O] # often O = C
            x = self.so3_linear_2(x)
        elif self.version == 3:
            x = self.so3_linear_2(x)

        # final linear layer without activation
        # [B, D, H] -> [B, D, O] # often O = C
        # x = self.so3_linear_2(x)

        return x


class TransBlockV2(torch.nn.Module):
    """

    Args:
        sphere_channels (int):      Number of spherical channels
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        output_channels (int):      Number of output channels

        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution

        SO3_rotation (list:SO3_Rotation): Class to calculate Wigner-D matrices and rotate embeddings
        mappingReduced (CoefficientMappingModule): Class to convert l and m indices once node embedding is rotated
        SO3_grid (SO3_grid):        Class used to convert from grid the spherical harmonic representations

        max_num_elements (int):     Maximum number of atomic numbers
        edge_channels_list (list:int):  List of sizes of invariant edge embedding. For example, [input_channels, hidden_channels, hidden_channels].
                                        The last one will be used as hidden size when `use_atom_edge_embedding` is `True`.
        use_atom_edge_embedding (bool): Whether to use atomic embedding along with relative distance for edge scalar features
        use_m_share_rad (bool):     Whether all m components within a type-L vector of one channel share radial function weights

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFN.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        ln_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh'])

        alpha_drop (float):         Dropout rate for attention weights
        path_drop (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN

        use_variational_alpha_drop (bool): If `True`, use variational dropout for attention weights (applies the same mask at every solver step)
        use_variational_proj_drop (bool): If `True`, use variational dropout for outputs of attention and FFN (applies the same mask at every solver step)
    """

    def __init__(
        self,
        sphere_channels,  # input?
        attn_hidden_channels,
        num_heads,
        attn_alpha_channels,
        attn_value_channels,
        ffn_hidden_channels,
        output_channels,
        lmax_list,
        mmax_list,
        SO3_rotation,
        mappingReduced,
        SO3_grid,
        max_num_elements,
        edge_channels_list,
        use_atom_edge_embedding=True,
        use_m_share_rad=False,
        attn_activation="silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        ln_type="rms_norm_sh",
        alpha_drop=0.0,
        path_drop=0.0,
        proj_drop=0.0,
        # added
        use_variational_alpha_drop=False,
        use_variational_path_drop=False,
        ln_norm="component",
        ln_affine=True,
        ln="pre",  # pre, post
        final_ln=False,
        noise_hidden_in={},
        noise_hidden_out={},
    ):
        super(TransBlockV2, self).__init__()

        max_lmax = max(lmax_list)

        assert ln in [
            "pp",
            "pre",
            "post",
        ], "ln must be 'pre', 'pp' or 'post' but got {}".format(ln)
        if ln in ['both']:
            ln = "pp"
        self.ln = ln
        self.final_ln = final_ln

        self.noise_in = RecurrentNoise(**noise_hidden_in)

        self.norm_1 = get_normalization_layer(
            ln_type,
            lmax=max_lmax,
            num_channels=sphere_channels,
            normalization=ln_norm,
            affine=ln_affine,
        )

        self.graph_attention = SO2EquivariantGraphAttention(
            sphere_channels=sphere_channels,
            hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            output_channels=sphere_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_rotation=SO3_rotation,
            mappingReduced=mappingReduced,
            SO3_grid=SO3_grid,
            max_num_elements=max_num_elements,
            edge_channels_list=edge_channels_list,
            use_atom_edge_embedding=use_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            use_gate_act=use_gate_act,
            use_sep_s2_act=use_sep_s2_act,
            alpha_drop=alpha_drop,
            # added
            use_variational_alpha_drop=use_variational_alpha_drop,
        )

        if use_variational_path_drop:
            # print(f"{self.__class__.__name__}: Using VariationalGraphPathDrop.")
            self.path_drop = (
                VariationalGraphPathDrop(path_drop) if path_drop > 0.0 else None
            )
        else:
            # print(f"{self.__class__.__name__}: Not using VariationalGraphPathDrop.")
            self.path_drop = GraphPathDrop(path_drop) if path_drop > 0.0 else None

        # TODO: what does this do? It's set to 0 in every config
        self.proj_drop = (
            EquivariantDropoutArraySphericalHarmonics(proj_drop, drop_graph=False)
            if proj_drop > 0.0
            else None
        )

        # post layer norm -> after FF/FF_shortcut -> potentially reduce sphere channels
        if self.ln in ["pp", "pre"]:
            self.norm_2_pre = get_normalization_layer(
                ln_type,
                lmax=max_lmax,
                num_channels=sphere_channels,
                normalization=ln_norm,
                affine=ln_affine,
            )
        if self.ln in ["pp", "post"]:
            self.norm_2_post = get_normalization_layer(
                ln_type,
                lmax=max_lmax,
                num_channels=output_channels,
                normalization=ln_norm,
                affine=ln_affine,
            )

        self.ffn = FeedForwardNetwork(
            sphere_channels=sphere_channels,
            hidden_channels=ffn_hidden_channels,
            output_channels=output_channels,
            lmax_list=lmax_list,
            mmax_list=mmax_list,
            SO3_grid=SO3_grid,
            activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,
        )

        # in =/= out
        if sphere_channels != output_channels:
            self.ffn_shortcut = SO3_LinearV2(
                sphere_channels, output_channels, lmax=max_lmax
            )
        else:
            self.ffn_shortcut = None
        
        if self.final_ln:
            self.norm_final = get_normalization_layer(
                ln_type,
                lmax=max_lmax,
                num_channels=output_channels,
                normalization=ln_norm,
                affine=ln_affine,
            )

        self.noise_out = RecurrentNoise(**noise_hidden_out)

    def forward(
        self,
        x,  # SO3_Embedding
        atomic_numbers,
        edge_distance,
        edge_index,
        batch,  # for GraphPathDrop
        **kwargs,
    ):
        """Norm, GraphAttention, PathDrop, ProjDrop, Norm, FFN, PathDrop, ProjDrop"""

        x.embedding = self.noise_in(x.embedding)

        output_embedding = x

        # Open residual connection
        x_res = output_embedding.embedding

        print_values(a=x_res, name="TansBlockIn")

        # Norm
        if self.ln in ["pp", "pre"]:
            output_embedding.embedding = self.norm_1(output_embedding.embedding)
        # GraphAttention
        output_embedding = self.graph_attention(
            output_embedding, atomic_numbers, edge_distance, edge_index
        )

        # PathDrop
        if self.path_drop is not None:
            output_embedding.embedding = self.path_drop(
                output_embedding.embedding, batch
            )
        # ProjDrop
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch
            )

        # Merge residual connection
        output_embedding.embedding = output_embedding.embedding + x_res

        if self.ln in ["pp", "post"]:
            output_embedding.embedding = self.norm_1(output_embedding.embedding)

        # Open residual connection
        x_res = output_embedding.embedding

        # Norm
        if self.ln in ["pp", "pre"]:
            output_embedding.embedding = self.norm_2_pre(output_embedding.embedding)

        # FeedForwardNetwork
        output_embedding = self.ffn(output_embedding)

        # PathDrop
        if self.path_drop is not None:
            output_embedding.embedding = self.path_drop(
                output_embedding.embedding, batch
            )
        # ProjDrop
        if self.proj_drop is not None:
            output_embedding.embedding = self.proj_drop(
                output_embedding.embedding, batch
            )

        # Shortcut: in =/= out -> reduce dimension of residual connection
        if self.ffn_shortcut is not None:
            shortcut_embedding = SO3_Embedding(
                0,
                output_embedding.lmax_list.copy(),
                self.ffn_shortcut.in_features,
                device=output_embedding.device,
                dtype=output_embedding.dtype,
            )
            shortcut_embedding.set_embedding(x_res)
            shortcut_embedding.set_lmax_mmax(
                output_embedding.lmax_list.copy(), output_embedding.lmax_list.copy()
            )
            shortcut_embedding = self.ffn_shortcut(shortcut_embedding)
            x_res = shortcut_embedding.embedding

        # Merge residual connection
        output_embedding.embedding = output_embedding.embedding + x_res

        # post layer norm
        # after shortcut, i.e. sphere channels are potentially reduced
        if self.ln in ["pp", "post"]:
            output_embedding.embedding = self.norm_2_post(output_embedding.embedding)
        if self.final_ln:
            output_embedding.embedding = self.norm_final(output_embedding.embedding)
        
        output_embedding.embedding = self.noise_out(output_embedding.embedding)

        return output_embedding
