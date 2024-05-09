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
import sys

# add the root of the project to the path so it can find equiformer
# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# print("root_dir:", root_dir)
# sys.path.append(root_dir)

from pathlib import Path
from typing import Iterable, Optional

from equiformer.logger import FileLogger
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

from equiformer_v2.nets.equiformer_v2.gaussian_rbf import GaussianRadialBasisLayer
from equiformer_v2.nets.equiformer_v2.edge_rot_mat import init_edge_rot_mat
from equiformer_v2.nets.equiformer_v2.so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2,
)
from equiformer_v2.nets.equiformer_v2.module_list import ModuleListInfo
from equiformer_v2.nets.equiformer_v2.so2_ops import SO2_Convolution
from equiformer_v2.nets.equiformer_v2.radial_function import RadialFunction
from equiformer_v2.nets.equiformer_v2.layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from equiformer_v2.nets.equiformer_v2.transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2,
)

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb

import matplotlib.pyplot as plt
import seaborn as sns

from typing import List

from equiformer_v2.oc20.trainer.base_trainer_oc20 import Normalizer

import deq2ff.register_all_models

# torch_geometric/data/collate.py:150: UserWarning: An output with one or more elements was resized since it had shape, which does not match the required output shape
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def myround(x):
    return round(x)


def deq_implicit_layer_wrapper(
    model, x, emb, edge_index, edge_distance, atomic_numbers, data, norms
):
    # ints are immutable but lists are mutable (so we can pass by reference and change inplace)
    x = model.deq_implicit_layer(
        x,
        emb=emb,
        edge_index=edge_index,
        edge_distance=edge_distance,
        atomic_numbers=atomic_numbers,
        data=data,
    )
    norms.append([x.norm().item(), x.norm(1).item(), x.norm(2).item()])
    print(
        f"x ({len(norms)}): fro={myround(x.norm().item())}, l1={myround(x.norm(1).item())}, l2={myround(x.norm(2).item())}"
    )
    return x


def main(
    args,
    weight_init="uniform",
    num_layers=2,
    cat_injection=False,
    norm_injection="prev",
    normlayer_norm="component",
    norm_type="rms_norm_sh",
):

    args.model.weight_init = weight_init
    args.model.num_layers = num_layers
    args.model.cat_injection = cat_injection
    args.model.norm_injection = norm_injection  # None, prev

    args.model.normlayer_norm = normlayer_norm  # component, norm
    args.model.norm_type = norm_type  # ['rms_norm_sh', 'layer_norm', 'layer_norm_sh']

    # pretty print args
    # print(OmegaConf.to_yaml(args))

    # since dataset needs random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """ Dataset """
    import equiformer.datasets.pyg.md_all as md_all

    train_dataset, val_dataset, test_dataset, all_dataset = md_all.get_md_datasets(
        root=args.data_path,
        dataset_arg=args.target,
        dname=args.dname,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        order=md_all.get_order(args),
    )

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    task_mean = float(y.mean())
    task_std = float(y.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Normalizers
    if args.normalizer == "md17":
        normalizer_e = lambda x: (x - task_mean) / task_std
        normalizer_f = lambda x: x / task_std
    elif args.normalizer == "oc20":
        normalizer_e = Normalizer(
            mean=task_mean,
            std=task_std,
            device=device,
        )
        normalizer_f = Normalizer(
            mean=0,
            std=task_std,
            device=device,
        )
    else:
        raise NotImplementedError(f"Unknown normalizer: {args.normalizer}")
    normalizers = {"energy": normalizer_e, "force": normalizer_f}

    """ Network """
    create_model = model_entrypoint(args.model.name)
    if "deq_kwargs" in args:
        model = create_model(
            task_mean=task_mean,
            task_std=task_std,
            **args.model,
            deq_kwargs=args.deq_kwargs,
        )
    else:
        model = create_model(task_mean=task_mean, task_std=task_std, **args.model)
    model = model.to(device)

    """ Data Loader """
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    """ Forward pass """

    first_batch = next(iter(train_loader))
    data = first_batch.to(device)

    # energy, force
    pred_y, pred_dy, info = model(
        data=data,  # for EquiformerV2
        node_atom=data.z,
        pos=data.pos,
        batch=data.batch,
        # step=global_step,
        # datasplit="train",
    )

    """ Test """
    self = model

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
    ) = self.generate_graph(data)

    ###############################################################
    # Initialize data structures
    ###############################################################

    # Compute 3x3 rotation matrix per edge
    # data unused
    edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)

    # Initialize the WignerD matrices and other values for spherical harmonic calculations
    for i in range(self.num_resolutions):
        self.SO3_rotation[i].set_wigner(edge_rot_mat)

    ###############################################################
    # Initialize node embeddings
    ###############################################################

    # Init per node representations using an atomic number based embedding
    offset = 0
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
    edge_degree = self.edge_degree_embedding(atomic_numbers, edge_distance, edge_index)
    # both: [num_atoms, num_coefficients, num_channels]
    # num_coefficients = sum([(2 * l + 1) for l in self.lmax_list])
    # addition, not concatenation
    x.embedding = x.embedding + edge_degree.embedding

    ###############################################################
    # Update spherical node embeddings
    # "Replaced" by DEQ
    ###############################################################

    # emb_SO3 = x
    emb = x.embedding

    x = self._init_z(shape=emb.shape)

    reset_norm(self.blocks)

    print(
        f"\n emb: fro={myround(emb.norm().item())}, l1={myround(emb.norm(1).item())}, l2={myround(emb.norm(2).item())}"
    )

    # Transformer blocks
    # f = lambda z: self.mfn_forward(z, u)
    norms = []
    f = lambda x: deq_implicit_layer_wrapper(
        self,
        x,
        emb=emb,
        edge_index=edge_index,
        edge_distance=edge_distance,
        atomic_numbers=atomic_numbers,
        data=data,
        norms=norms,
    )

    # find fixed-point
    # solver_kwargs = {"f_max_iter": 0} if reuse else {} # TODO
    solver_kwargs = {}
    # returns the sampled fixed point trajectory (tracked gradients)
    # z_pred, info = self.deq(f, z, solver_kwargs=solver_kwargs)
    z_pred, info = self.deq(f, x, solver_kwargs=solver_kwargs)

    x = SO3_Embedding(
        length=num_atoms,
        lmax_list=self.lmax_list,
        num_channels=self.sphere_channels_fixedpoint,
        device=self.device,
        dtype=self.dtype,
        embedding=z_pred[-1],
    )

    # Final layer norm
    x.embedding = self.norm(x.embedding)

    return norms


def plot_norms_layer(args):
    sns.set_style("whitegrid")
    colors = sns.color_palette()

    # create two figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    i = 0
    for norm_type in ["layer_norm_sh", "layer_norm", "rms_norm_sh"]:
        for normlayer_norm in ["component", "norm"]:
            norms = main(
                args,
                norm_injection="prev",
                norm_type=norm_type,
                normlayer_norm=normlayer_norm,
            )
            norms = np.array(norms)

            # plot
            ax1.plot(
                norms[:, 0], label=f"{norm_type} {normlayer_norm}", color=colors[i]
            )
            ax2.plot(
                norms[:, 1],
                label=f"{norm_type} {normlayer_norm}",
                color=colors[i],
                linestyle="--",
            )
            i += 1

    ax1.title.set_text("l2 norm")
    ax2.title.set_text("l1 norm")
    plt.legend()
    plt.xlabel("forward passes through implicit layer")
    plt.ylabel("norm of node embedding")
    fpath = "figs/layernorm_fsolver.png"
    plt.savefig(fpath)
    print(f"{fpath} saved")
    return


def plot_input_inj(args):
    sns.set_style("whitegrid")
    colors = sns.color_palette()

    # create two figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    i = 0
    for cat_injection in [True, False]:
        # for norm_injection in [None, 'one', 'prev']:
        for norm_injection in ["one", "prev"]:
            norms = main(
                args,
                cat_injection=cat_injection,
                norm_injection=norm_injection,
                norm_type="rms_norm_sh",
                normlayer_norm="norm",
            )
            norms = np.array(norms)

            # plot
            cat = "cat" if cat_injection else "add"
            _norm = "" if cat_injection else norm_injection
            ax1.plot(norms[:, 0], label=f"{cat} {_norm}", color=colors[i])
            ax2.plot(
                norms[:, 1], label=f"{cat} {_norm}", color=colors[i], linestyle="--"
            )
            i += 1

            if cat_injection:
                break

    ax1.title.set_text("l2 norm")
    ax2.title.set_text("l1 norm")
    plt.legend()
    plt.xlabel("forward passes through implicit layer")
    plt.ylabel("norm")
    fpath = "figs/inputinjection_norm.png"
    plt.savefig(fpath)
    print(f"{fpath} saved")
    return


@hydra.main(
    config_name="md17", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""
    args.wandb = False

    # load deq config
    deq_config = OmegaConf.load("equiformer_v2/config/use/deq.yaml")
    # update args
    args = OmegaConf.to_container(args, resolve=True)
    deq_config = OmegaConf.to_container(deq_config, resolve=True)
    for k, v in deq_config.items():
        # add
        if k not in args:
            args[k] = v
        elif type(v) == dict:
            # v is a dict
            # update
            for kk, vv in v.items():
                if kk not in args:
                    args[k][kk] = vv
                elif type(vv) == dict:
                    # v is a dict in a dict
                    for kkk, vvv in vv.items():
                        args[k][kk][kkk] = vvv
                else:
                    args[k][kk] = vv
        else:
            # no nesting
            # add or update
            args[k] = v

    # pretty print args
    # import yaml
    # print(yaml.dump(args))

    # make args to DictConfig again
    args = OmegaConf.create(args)

    from deq2ff.logging_utils import init_wandb

    # init_wandb(args, project="equilibrium-forcefields-equiformer_v2")
    init_wandb(args)

    # norms = main(args, norm_injection='prev', norm_type='layer_norm_sh', normlayer_norm='component')

    # plot_norms_layer(args)
    plot_input_inj(args)


if __name__ == "__main__":
    hydra_wrapper()
