import torch

import torchdeq
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm

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

from equiformer.oc20.trainer.logger import FileLogger
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

from deq2ff.losses import load_loss, L2MAELoss, _pairwise_distances

# registers all models
import deq2ff.register_all_models

# torch_geometric/data/collate.py:150: UserWarning: An output with one or more elements was resized since it had shape, which does not match the required output shape
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def myround(x):
    return round(x)


"""
What happens if we set f_max_iter=0?
All steps are done with autograd?


"""


def run_deq(deq, f, theta, name=""):
    z0 = torch.tensor(0.0)
    z_out, info = deq(f, z0)

    print(f"nstep", info["nstep"])

    tgt = torch.tensor(0.5)
    loss = (z_out[-1] - tgt).abs().mean()
    loss.backward()

    print(f"Loss & Grad {name}:", loss.item(), theta.grad)


def test_if_broyden_is_differentiable():
    # set everything to GPU
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_default_dtype(torch.float32)
    torch.set_default_device("cuda")

    # """ Define function """
    # # Input Injection & Equilibrium function
    # x = torch.tensor(1.0)
    # theta = torch.tensor(0.0, requires_grad=True)
    # f = lambda z: torch.cos(z) + theta

    # """ 1-step grad """
    # print("\n1-step grad")
    # # broyden, fixed_point_iter
    # deq = get_deq(f_solver='fixed_point_iter', f_max_iter=20)
    # run_deq(deq, f, theta)

    # """ BPTT, 10 steps """
    # print("\nBPTT, 10 steps")
    # deq = get_deq(grad=10, f_max_iter=0)
    # run_deq(deq, f, theta)

    # """ BPTT, 100 steps """
    # print("\nBPTT, 100 steps")
    # deq = get_deq(grad=100, f_max_iter=0)
    # run_deq(deq, f, theta)

    """ Broyden with gradients """
    print("\nBroyden with gradients")
    from torchdeq.solver.broyden import broyden_solver, broyden_solver_grad

    torch.autograd.set_detect_anomaly(True)

    theta = torch.tensor(0.0, requires_grad=True)
    f = lambda z: torch.cos(z) + theta

    z0 = torch.tensor(0.0)
    z0.requires_grad = True
    # This will only work for max_iter=1
    z_out, _, info = broyden_solver_grad(
        f, z0, max_iter=4, tol=1e-3, return_final=False
    )
    # z_out = [z_out]

    tgt = torch.tensor(0.5)
    loss = (z_out - tgt).abs().mean()
    loss.backward()

    print(f"Loss & Grad:", loss.item(), theta.grad)
    print(f"nstep", info["nstep"])


def test_if_differentiable_broyden_same_as_before(args):
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
        test_patch_size=None,
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

    loss_fn = load_loss({"energy": args.loss_energy, "force": args.loss_force})
    criterion_energy = loss_fn["energy"]
    criterion_force = loss_fn["force"]

    """ Data Loader """
    shuffle = True
    if args.datasplit in ["fpreuse_ordered"]:
        shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    """Train for one epoch"""
    data_loader = train_loader
    global_step = 0

    model.train()
    # criterion.train()
    criterion_energy.train()
    criterion_force.train()

    task_mean = model.task_mean
    task_std = model.task_std

    from torchdeq.solver.broyden import test_broyden_solver_grad

    for step, data in enumerate(data_loader):
        print()
        data = data.to(device)

        solver_kwargs = {}

        with torch.no_grad():
            # energy, force

            # copy model "encoder"
            self = model
            fixedpoint = None

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
                    x.embedding[:, offset_res, :] = self.sphere_embedding(
                        atomic_numbers
                    )
                else:
                    x.embedding[:, offset_res, :] = self.sphere_embedding(
                        atomic_numbers
                    )[:, offset : offset + self.sphere_channels]
                offset = offset + self.sphere_channels
                offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

            # Edge encoding (distance and atom edge)
            edge_distance = self.distance_expansion(edge_distance)
            if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
                source_element = atomic_numbers[
                    edge_index[0]
                ]  # Source atom atomic number
                target_element = atomic_numbers[
                    edge_index[1]
                ]  # Target atom atomic number
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

            # if self.learn_scale_after_encoder:
            x.embedding = x.embedding * self.learn_scale_after_encoder

            ###############################################################
            # Update spherical node embeddings
            # "Replaced" by DEQ
            ###############################################################

            # emb_SO3 = x
            emb = x.embedding

            if fixedpoint is None:
                x: torch.Tensor = self._init_z(shape=emb.shape, emb=emb)
                reuse = False
            else:
                reuse = True
                x = fixedpoint

            reset_norm(self.blocks)

            # Transformer blocks
            # f = lambda z: self.mfn_forward(z, u)
            def f(x):
                return self.deq_implicit_layer(
                    x,
                    emb=emb,
                    edge_index=edge_index,
                    edge_distance=edge_distance,
                    atomic_numbers=atomic_numbers,
                    data=data,
                )

            z_pred, _, info = test_broyden_solver_grad(f, x)

        exit()

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

    # init_wandb(args, project="oc20-ev2")
    init_wandb(args)

    test_if_differentiable_broyden_same_as_before(args)


if __name__ == "__main__":
    hydra_wrapper()
