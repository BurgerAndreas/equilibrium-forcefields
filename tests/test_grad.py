"""
Test the differences between inexact and exact gradients.
- The gradient between grad=1, grad=3 to ift=True should not be too different.
- The force should be the same.

When testing the same basemodel (equiformer):
    -> energy, force, loss_e are about e-7 different (rounding errors)
    -> loss_f = 0 (rounding errors)
"""

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
from pathlib import Path
from typing import Iterable, Optional

import sys, os

# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import equiformer.datasets.pyg.md17 as md17_dataset

from equiformer.oc20.trainer.logger import FileLogger

# import equiformer.nets
from equiformer.nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from equiformer.optim_factory import create_optimizer

from equiformer.engine import AverageMeter, compute_stats

import hydra
import wandb
import omegaconf
from omegaconf import DictConfig, OmegaConf
import copy

from equiformer.main_md17 import L2MAELoss

import deq2ff
from deq2ff.deq_equiformer.deq_dp_md17 import *
from deq2ff.deq_equiformer.deq_graph_md17 import *

ModelEma = ModelEmaV2


def difference(a, b):
    return torch.mean(torch.abs(a - b))


def absscale(a):
    return torch.mean(torch.abs(a))


import warnings

# /torch/jit/_check.py:177: UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`.
# Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
# warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

# load args from hydra config
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
config_dir = os.path.join(root_dir, "equiformer/config")


@hydra.main(config_name="deq", config_path=config_dir, version_base="1.3")
def do_test(args_base: DictConfig) -> None:
    # https://github.com/locuslab/torchdeq/blob/4f6bd5fa66dd991cad74fcc847c88061764cf8db/torchdeq/grad.py#L155
    deq_kwargs_sweep = [
        {"grad": 1},
        {"grad": 3},
        {"grad": 10},
        {"ift": True},
    ]

    # test the same basemodel (equiformer) for rounding errors
    # deq_kwargs_sweep = [
    #     {}, {},
    # ]

    args_base = OmegaConf.structured(OmegaConf.to_yaml(args_base))
    # args_base = omegaconf.OmegaConf.resolve(args_base)
    # args_base = OmegaConf.to_container(args_base, resolve=True) # convert to dict

    print("args_base:\n", args_base)

    log_energy = []
    log_force = []
    log_loss_e = []
    log_loss_f = []
    log_grad = []

    prev_input = None
    prev_nn_weights = None

    print("")
    for deq_kwargs in deq_kwargs_sweep:

        args = args_base.copy()
        for k, v in deq_kwargs.items():
            args[k] = v

        # basemodel or deq
        if len(deq_kwargs) == 0:
            args.model_is_deq = False
        if args.model_is_deq == True:
            args.model.name = f"deq_{args.model.name}"
        print(f"\nmodel_name: {args.model.name}")

        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        """ Dataset """
        train_dataset, val_dataset, test_dataset = md17_dataset.get_md17_datasets(
            root=os.path.join(args.data_path, "md17", args.target),
            dataset_arg=args.target,
            train_size=args.train_size,
            val_size=args.val_size,
            test_patch_size=None,
            seed=args.seed,
        )

        # statistics
        y = torch.cat([batch.y for batch in train_dataset], dim=0)
        mean = float(y.mean())
        std = float(y.std())

        # since dataset needs random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """ Network """
        create_model = model_entrypoint(args.model.name)
        if args.model_is_deq:
            model = create_model(
                irreps_in=args.input_irreps,
                radius=args.radius,
                number_of_basis=args.number_of_basis,
                task_mean=mean,
                task_std=std,
                atomref=None,
                path_drop=args.path_drop,
                num_layers=args.num_layers,
                deq_kwargs=args.deq_kwargs,
            )
        else:
            model = create_model(
                irreps_in=args.input_irreps,
                radius=args.radius,
                number_of_basis=args.number_of_basis,
                task_mean=mean,
                task_std=std,
                atomref=None,
                path_drop=args.path_drop,
                num_layers=args.num_layers,
            )
        print(f"model parameters: {sum(p.numel() for p in model.parameters())}")

        try:
            if prev_nn_weights is not None:
                model.load_state_dict(prev_nn_weights)
        except Exception as e:
            print(f" prev_nn_weights: {prev_nn_weights.keys()}")
            raise e
        prev_nn_weights = copy.deepcopy(model.state_dict())

        # if args.checkpoint_path is not None:
        #     state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        #     model.load_state_dict(state_dict["state_dict"])
        # else:
        #     print("Random initialization!")

        model = model.to(device)

        """ Optimizer and LR Scheduler """
        optimizer = create_optimizer(args, model)
        lr_scheduler, _ = create_scheduler(args, optimizer)
        criterion = (
            L2MAELoss()
        )  # torch.nn.L1Loss()  #torch.nn.MSELoss() # torch.nn.L1Loss()

        """ Data Loader """
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

        global_step = 0
        start_time = time.perf_counter()
        # for epoch in range(args.epochs):

        epoch_start_time = time.perf_counter()

        lr_scheduler.step(0)

        # train_one_epoch
        model.train()
        criterion.train()

        loss_metrics = {"energy": AverageMeter(), "force": AverageMeter()}
        mae_metrics = {"energy": AverageMeter(), "force": AverageMeter()}

        start_time = time.perf_counter()

        task_mean = model.task_mean
        task_std = model.task_std

        # z_star = None
        for step, data in enumerate(train_loader):

            # torch_geometric.data.batch.DataBatch
            data = data.to(device)

            # check that the data is the same
            if prev_input is not None:
                assert torch.allclose(
                    prev_input, data.z
                ), f"{difference(prev_input, data.z)}"
            prev_input = data.z

            # energy, force
            pred_y, pred_dy = model(node_atom=data.z, pos=data.pos, batch=data.batch)
            # if deq_mode and reuse:
            #     z_star = z_pred.detach()

            loss_e = criterion(pred_y, ((data.y - task_mean) / task_std))
            loss = args.energy_weight * loss_e

            loss_f = criterion(pred_dy, (data.dy / task_std))
            loss += args.force_weight * loss_f

            # log
            log_energy.append(pred_y.detach())
            log_force.append(pred_dy.detach())
            log_loss_e.append(loss_e.detach())
            log_loss_f.append(loss_f.detach())
            # log_grad.append(args.grad)

            break

    # check
    for i in range(len(log_energy)):
        for j in range(i + 1, len(log_energy)):
            print("")
            print(f" {deq_kwargs_sweep[i]}  |  {deq_kwargs_sweep[j]}")
            if deq_kwargs_sweep[i] == deq_kwargs_sweep[j]:
                print(
                    "Same model -> difference is due to rounding errors or randomness (e-7)"
                )
            print(
                f"energy difference: {difference(log_energy[i], log_energy[j])}   (magnitude: {absscale(log_energy[i])})"
            )
            print(
                f"force difference:  {difference(log_force[i], log_force[j])}   (magnitude: {absscale(log_force[i])})"
            )
            print(
                f"loss_e difference: {difference(log_loss_e[i], log_loss_e[j])}   (magnitude: {absscale(log_loss_e[i])})"
            )
            print(
                f"loss_f difference: {difference(log_loss_f[i], log_loss_f[j])}   (magnitude: {absscale(log_loss_f[i])})"
            )
            # print(f'grad: {log_grad[i] == log_grad[j]}')


if __name__ == "__main__":
    do_test()
