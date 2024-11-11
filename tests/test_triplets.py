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

from deq2ff.losses import (
    load_loss,
    L2MAELoss,
    _pairwise_distances,
    TripletLoss,
    calc_triplet_loss,
    TripletDataloader,
)

# registers all models
import deq2ff.register_all_models

# torch_geometric/data/collate.py:150: UserWarning: An output with one or more elements was resized since it had shape, which does not match the required output shape
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def myround(x):
    return round(x)


class DummyClass:
    def __init__(self, natoms=2, batch_size=4):
        self.natoms = torch.full((batch_size,), fill_value=natoms)


def fake_fixedpoints(batch_size, natoms, z=None):
    if z is None:
        z = torch.arange(batch_size)
    fixedpoints = z.repeat_interleave(natoms).unsqueeze(1).float()
    fixedpoints = fixedpoints.expand(-1, 3)
    # fixedpoints = fixedpoints.expand(args.batch_size*natoms, 3, 3)
    return fixedpoints


def test_triplet_simple():
    args = omegaconf.OmegaConf.create()
    args.batch_size = 4
    args.tripletloss_margin = 1.0

    # dummy dataset
    dataset = torch.arange(1000)

    dataloader = TripletDataloader(dataset, args.batch_size)
    for i, data in enumerate(dataloader):
        print()
        print(f"Batch {i}: {data}")
        if i > 0:
            break

        # fake fixedpoints
        natoms = 2
        fixedpoints = fake_fixedpoints(args.batch_size, natoms)
        # fake data
        data = DummyClass(natoms=natoms, batch_size=args.batch_size)

        print("fixedpoints", fixedpoints.shape)

        triplet_lossfn = TripletLoss(margin=args.tripletloss_margin)
        loss = calc_triplet_loss(fixedpoints, data, triplet_lossfn)
        print("loss", loss)


def main(args):

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

    for step, data in enumerate(data_loader):
        print()
        data = data.to(device)

        exit()

    # print("train_dataset", train_dataset)


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

    args.datasplit = "fpreuse_ordered"

    main(args)


if __name__ == "__main__":

    test_triplet_simple()

    # hydra_wrapper()
