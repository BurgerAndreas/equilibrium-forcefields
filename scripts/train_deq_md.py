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

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb

"""
Adapted from equiformer/main_md17.py
"""

import deq2ff
from deq2ff.deq_equiformer.deq_dp_md17 import (
    deq_dot_product_attention_transformer_exp_l2_md17,
)
from deq2ff.deq_equiformer.deq_graph_md17 import (
    deq_graph_attention_transformer_nonlinear_l2_md17,
)
from deq2ff.deq_equiformer.deq_dp_md17_noforce import (
    deq_dot_product_attention_transformer_exp_l2_md17_noforce,
)
# DEQ EquiformerV2
# from deq2ff.deq_equiformer_v2.deq_equiformer_v2_oc20 import (
#     deq_equiformer_v2_oc20,
# )

ModelEma = ModelEmaV2

# silence:
# UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`.
# Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/modules/loss.py#L7
class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


def get_force_placeholder(dy, loss_e):
    """if meas_force is False, return a placeholder for force prediction and loss_f"""
    # pred_dy = torch.zeros_like(data.dy)
    pred_dy = torch.full_like(dy, float("nan"))
    # loss_f = torch.zeros_like(loss_e)
    loss_f = torch.full_like(loss_e, float("nan"))
    return pred_dy, loss_f


def main(args):

    # create output directory
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(
        f"args passed to {__file__} main():\n {omegaconf.OmegaConf.to_yaml(args)}"
    )

    # since dataset needs random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """ Dataset """
    if args.use_original_datasetcreation:
        import equiformer.datasets.pyg.md17 as md17_dataset

        train_dataset, val_dataset, test_dataset = md17_dataset.get_md17_datasets(
            root=os.path.join(args.data_path, "md17", args.target),
            dataset_arg=args.target,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=None,
            seed=args.seed,
            # order="consecutive_test" if args.fpreuse_test else None,
        )
    else:
        import equiformer.datasets.pyg.md_all as md_all

        order = None
        if ('fpreuse_test' in args and args.fpreuse_test) or ('fpreuse_datasplit' in args and args.fpreuse_datasplit):
            order = "consecutive_test"

        train_dataset, val_dataset, test_dataset = md_all.get_md_datasets(
            root=args.data_path,
            dataset_arg=args.target,
            dname=args.dname,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=None,
            seed=args.seed,
            load_splits=args.use_revised_splits,
            order=order,
        )

    _log.info("")
    _log.info("Training set size:   {}".format(len(train_dataset)))
    _log.info("Validation set size: {}".format(len(val_dataset)))
    _log.info("Testing set size:    {}\n".format(len(test_dataset)))

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())
    _log.info("Training set mean: {}, std: {}\n".format(mean, std))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Network """
    create_model = model_entrypoint(args.model.name)
    if "deq_kwargs" in args:
        model = create_model(
            task_mean=mean,
            task_std=std,
            **args.model,
            deq_kwargs=args.deq_kwargs,
        )
    else:
        model = create_model(task_mean=mean, task_std=std, **args.model)
    _log.info(
        f"model {args.model.name} created with kwargs \n{omegaconf.OmegaConf.to_yaml(args.model)}"
    )
    _log.info(f"Model: \n{model}")

    # watch gradients, weights, and activations
    # https://docs.wandb.ai/ref/python/watch
    if args.watch_model:
        wandb.watch(model, log="all", log_freq=args.log_every_step_major)

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict["state_dict"])
        _log.info(f"Loaded model from {args.checkpoint_path}")

    model = model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info("Number of Model params: {}".format(n_parameters))
    wandb.run.summary["Model Parameters"] = n_parameters
    # wandb.config.update({"Model Parameters": n_parameters})

    # parameters in transformer blocks / deq implicit layers
    n_parameters = sum(p.numel() for p in model.blocks.parameters() if p.requires_grad)
    _log.info("Number of DEQLayer params: {}".format(n_parameters))
    wandb.run.summary["DEQLayer Parameters"] = n_parameters

    # decoder
    try:
        n_parameters = sum(
            p.numel() for p in model.final_block.parameters() if p.requires_grad
        )
        _log.info("Number of FinalBlock params: {}".format(n_parameters))
        wandb.run.summary["FinalBlock Parameters"] = n_parameters
    except:
        print(f"AttributeError: '{model.__class__.__name__}' object has no attribute 'final_block'")

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
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # added drop_last=True to avoid error with fixed-point reuse
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True
    )
    if args.fpreuse_test:
        # reorder test dataset to be consecutive
        from deq2ff.data_utils import reorder_dataset

        test_dataset = reorder_dataset(test_dataset, args.eval_batch_size)
        _log.info(f"Reordered test dataset to be consecutive for fixed-point reuse.")
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True
    )

    """ Compute stats """
    # Compute _AVG_NUM_NODES, _AVG_DEGREE
    if args.compute_stats:
        avg_node, avg_edge, avg_degree = compute_stats(
            train_loader,
            max_radius=args.max_radius,
            logger=_log,
            print_freq=args.print_freq,
        )
        return

    # record the best validation and testing errors and corresponding epochs
    best_metrics = {
        "val_epoch": 0,
        "test_epoch": 0,
        "val_force_err": float("inf"),
        "val_energy_err": float("inf"),
        "test_force_err": float("inf"),
        "test_energy_err": float("inf"),
    }
    best_ema_metrics = {
        "val_epoch": 0,
        "test_epoch": 0,
        "val_force_err": float("inf"),
        "val_energy_err": float("inf"),
        "test_force_err": float("inf"),
        "test_energy_err": float("inf"),
    }

    global_step = 0

    # dryrun (tryrun) for logging
    try:
        model.train()
        criterion.train()

        for step, data in enumerate(train_loader):
            data = data.to(device)

            # energy, force
            shapes_to_log = model.dummy_forward_for_logging(
                node_atom=data.z,
                pos=data.pos,
                batch=data.batch,
            )
            break

        # nums include batch size
        shapes_to_log["batch_size"] = args.batch_size
        shapes_to_log["NumNodes"] = shapes_to_log["NumNodes"] // args.batch_size
        shapes_to_log["NumEdges"] = shapes_to_log["NumEdges"] // args.batch_size

        import pprint

        ppr = pprint.PrettyPrinter(indent=4)
        ppr.pprint(shapes_to_log)
        wandb.run.summary.update(shapes_to_log)

        if args.dummy_logging_only:
            return True

        success = True
    except Exception as e:
        success = False
        _log.info(f"Error: {e}")

    if args.evaluate:
        test_err, test_loss = evaluate(
            args=args,
            model=model,
            criterion=criterion,
            data_loader=test_loader,
            device=device,
            print_freq=args.print_freq,
            logger=_log,
            print_progress=True,
            max_iter=-1,
            global_step=global_step,
            datasplit="test",
        )
        return

    _log.info("\nStart training!\n")
    start_time = time.perf_counter()
    for epoch in range(args.epochs):

        epoch_start_time = time.perf_counter()

        lr_scheduler.step(epoch)

        train_err, train_loss, global_step = train_one_epoch(
            args=args,
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            model_ema=model_ema,
            print_freq=args.print_freq,
            logger=_log,
            meas_force=args.meas_force,
        )

        optimizer.zero_grad()
        val_err, val_loss = evaluate(
            args=args,
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            optimizer=optimizer,
            device=device,
            print_freq=args.print_freq,
            logger=_log,
            print_progress=False,
            max_iter=-1,
            global_step=global_step,
            datasplit="val",
        )
        optimizer.zero_grad()

        if (epoch + 1) % args.test_interval == 0:
            _log.info(f"Testing model at epoch {epoch}")
            test_err, test_loss = evaluate(
                args=args,
                model=model,
                criterion=criterion,
                data_loader=test_loader,
                optimizer=optimizer,
                device=device,
                print_freq=args.print_freq,
                logger=_log,
                print_progress=True,
                max_iter=args.test_max_iter,
                global_step=global_step,
                datasplit="test",
            )
            optimizer.zero_grad()
        else:
            test_err, test_loss = None, None

        update_val_result, update_test_result = update_best_results(
            args, best_metrics, val_err, test_err, epoch
        )
        if update_val_result and args.save_best_checkpoint:
            _log.info(f"Saving best val checkpoint")
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(
                    args.output_dir,
                    "best_val_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, val_err["energy"].avg, val_err["force"].avg
                    ),
                ),
            )
        if update_test_result and args.save_best_checkpoint:
            _log.info(f"Saving best test checkpoint")
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(
                    args.output_dir,
                    "best_test_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, test_err["energy"].avg, test_err["force"].avg
                    ),
                ),
            )
        if (
            (epoch + 1) % args.test_interval == 0
            and (not update_val_result)
            and (not update_test_result)
            and args.save_periodic_checkpoint
        ):
            _log.info(f"Saving checkpoint")
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(
                    args.output_dir,
                    "epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, test_err["energy"].avg, test_err["force"].avg
                    ),
                ),
            )

        # log once per epoch
        info_str = "Epoch: [{epoch}] Target: [{target}] train_e_MAE: {train_e_mae:.5f}, train_f_MAE: {train_f_mae:.5f}, ".format(
            epoch=epoch,
            target=args.target,
            train_e_mae=train_err["energy"].avg,
            train_f_mae=train_err["force"].avg,
        )
        info_str += "val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, ".format(
            val_err["energy"].avg, val_err["force"].avg
        )
        if (epoch + 1) % args.test_interval == 0:
            info_str += "test_e_MAE: {:.5f}, test_f_MAE: {:.5f}, ".format(
                test_err["energy"].avg, test_err["force"].avg
            )
        info_str += "Time: {:.2f}s".format(time.perf_counter() - epoch_start_time)
        _log.info(info_str)

        # log to wandb
        logs = {
            "train_e_mae": train_err["energy"].avg,
            "train_f_mae": train_err["force"].avg,
            "val_e_mae": val_err["energy"].avg,
            "val_f_mae": val_err["force"].avg,
            "lr": optimizer.param_groups[0]["lr"],
            # allows us to plot against epoch
            # in the custom plots, click edit and select a custom x-axis
            "epoch": epoch,
            "time_train": time.perf_counter() - start_time,
            "time_per_epoch": time.perf_counter() - epoch_start_time,
        }
        if test_err is not None:
            logs["test_e_mae"] = test_err["energy"].avg
            logs["test_f_mae"] = test_err["force"].avg
        # if global_step % args.log_every_step_minor == 0:
        wandb.log(
            logs,
            # step=epoch,
            step=global_step,
        )

        info_str = "Best -- val_epoch={}, test_epoch={}, ".format(
            best_metrics["val_epoch"], best_metrics["test_epoch"]
        )
        info_str += "val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, ".format(
            best_metrics["val_energy_err"], best_metrics["val_force_err"]
        )
        info_str += "test_e_MAE: {:.5f}, test_f_MAE: {:.5f}\n".format(
            best_metrics["test_energy_err"], best_metrics["test_force_err"]
        )
        _log.info(info_str)

        # if global_step % args.log_every_step_major == 0:
        wandb.log(
            {
                "best_val_e_mae": best_metrics["val_energy_err"],
                "best_val_f_mae": best_metrics["val_force_err"],
                "best_test_e_mae": best_metrics["test_energy_err"],
                "best_test_f_mae": best_metrics["test_force_err"],
            },
            # step=epoch,
            step=global_step,
        )

        # evaluation with EMA
        if model_ema is not None:
            ema_val_err, _ = evaluate(
                args=args,
                model=model_ema.module,
                criterion=criterion,
                data_loader=val_loader,
                optimizer=optimizer,
                device=device,
                print_freq=args.print_freq,
                logger=_log,
                print_progress=False,
                global_step=global_step,
                datasplit="ema_val",
            )
            optimizer.zero_grad()

            if (epoch + 1) % args.test_interval == 0:
                _log.info(f"Testing EMA model at epoch {epoch}")
                ema_test_err, _ = evaluate(
                    args=args,
                    model=model_ema.module,
                    criterion=criterion,
                    data_loader=test_loader,
                    optimizer=optimizer,
                    device=device,
                    print_freq=args.print_freq,
                    logger=_log,
                    print_progress=True,
                    max_iter=args.test_max_iter,
                    global_step=global_step,
                    datasplit="ema_test",
                )
                optimizer.zero_grad()
            else:
                ema_test_err, ema_test_loss = None, None

            update_val_result, update_test_result = update_best_results(
                args, best_ema_metrics, ema_val_err, ema_test_err, epoch
            )

            if update_val_result and args.save_best_checkpoint:
                _log.info(f"Saving best EMA val checkpoint")
                torch.save(
                    {"state_dict": get_state_dict(model_ema)},
                    os.path.join(
                        args.output_dir,
                        "best_ema_val_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                            epoch, ema_val_err["energy"].avg, ema_val_err["force"].avg
                        ),
                    ),
                )
            if update_test_result and args.save_best_checkpoint:
                _log.info(f"Saving best EMA test checkpoint")
                torch.save(
                    {"state_dict": get_state_dict(model_ema)},
                    os.path.join(
                        args.output_dir,
                        "best_ema_test_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                            epoch, ema_test_err["energy"].avg, ema_test_err["force"].avg
                        ),
                    ),
                )
            if (
                (epoch + 1) % args.test_interval == 0
                and (not update_val_result)
                and (not update_test_result)
                and args.save_periodic_checkpoint
            ):
                _log.info(f"Saving EMA checkpoint")
                torch.save(
                    {"state_dict": get_state_dict(model_ema)},
                    os.path.join(
                        args.output_dir,
                        "ema_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                            epoch, test_err["energy"].avg, test_err["force"].avg
                        ),
                    ),
                )

            info_str = "EMA "
            info_str += "val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, ".format(
                ema_val_err["energy"].avg, ema_val_err["force"].avg
            )
            wandb.log(
                {
                    "EMA_val_e_mae": ema_val_err["energy"].avg,
                    "EMA_val_f_mae": ema_val_err["force"].avg,
                },
                # step=epoch,
                step=global_step,
            )

            if (epoch + 1) % args.test_interval == 0:
                info_str += "test_e_MAE: {:.5f}, test_f_MAE: {:.5f}, ".format(
                    ema_test_err["energy"].avg, ema_test_err["force"].avg
                )
                wandb.log(
                    {
                        "EMA_test_e_mae": ema_test_err["energy"].avg,
                        "EMA_test_f_mae": ema_test_err["force"].avg,
                    },
                    # step=epoch,
                    step=global_step,
                )

            info_str += "Time: {:.2f}s".format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)

            info_str = "Best EMA -- val_epoch={}, test_epoch={}, ".format(
                best_ema_metrics["val_epoch"], best_ema_metrics["test_epoch"]
            )
            info_str += "val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, ".format(
                best_ema_metrics["val_energy_err"], best_ema_metrics["val_force_err"]
            )
            info_str += "test_e_MAE: {:.5f}, test_f_MAE: {:.5f}\n".format(
                best_ema_metrics["test_energy_err"], best_ema_metrics["test_force_err"]
            )
            _log.info(info_str)

            # log to wandb
            wandb.log(
                {
                    "EMA_best_val_e_mae": best_ema_metrics["val_energy_err"],
                    "EMA_best_val_f_mae": best_ema_metrics["val_force_err"],
                    "EMA_best_test_e_mae": best_ema_metrics["test_energy_err"],
                    "EMA_best_test_f_mae": best_ema_metrics["test_force_err"],
                },
                # step=epoch,
                step=global_step,
            )

        # epoch done

    _log.info("\nAll epochs done!\nFinal test:")

    # all epochs done
    # evaluate on the whole testing set
    optimizer.zero_grad()
    test_err, test_loss = evaluate(
        args=args,
        model=model,
        criterion=criterion,
        data_loader=test_loader,
        optimizer=optimizer,
        device=device,
        print_freq=args.print_freq,
        logger=_log,
        print_progress=True,
        max_iter=args.test_max_iter_final,  # -1 means evaluate the whole dataset
        global_step=global_step,
        datasplit="test_final",
    )
    optimizer.zero_grad()

    # save the final model
    if args.save_final_checkpoint:
        _log.info(f"Saving final checkpoint")
        torch.save(
            {"state_dict": model.state_dict()},
            os.path.join(
                args.output_dir,
                "final_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                    epoch, test_err["energy"].avg, test_err["force"].avg
                ),
            ),
        )

    _log.info(f"Done!")
    _log.info(
        f"Final test error: MAE_e={test_err['energy'].avg}, MAE_f={test_err['force'].avg}"
    )
    return True


def update_best_results(args, best_metrics, val_err, test_err, epoch):
    def _compute_weighted_error(args, energy_err, force_err):
        return args.energy_weight * energy_err + args.force_weight * force_err

    update_val_result, update_test_result = False, False

    # _log.info(f"Trying to update best results for epoch {epoch}")
    new_loss = _compute_weighted_error(
        args, val_err["energy"].avg, val_err["force"].avg
    )
    prev_loss = _compute_weighted_error(
        args, best_metrics["val_energy_err"], best_metrics["val_force_err"]
    )
    if new_loss < prev_loss:
        best_metrics["val_energy_err"] = val_err["energy"].avg
        best_metrics["val_force_err"] = val_err["force"].avg
        best_metrics["val_epoch"] = epoch
        update_val_result = True

    if test_err is None:
        return update_val_result, update_test_result

    new_loss = _compute_weighted_error(
        args, test_err["energy"].avg, test_err["force"].avg
    )
    prev_loss = _compute_weighted_error(
        args, best_metrics["test_energy_err"], best_metrics["test_force_err"]
    )
    # _log.info(f" New loss test: {new_loss}, prev loss: {prev_loss}")
    if new_loss < prev_loss:
        best_metrics["test_energy_err"] = test_err["energy"].avg
        best_metrics["test_force_err"] = test_err["force"].avg
        best_metrics["test_epoch"] = epoch
        update_test_result = True

    return update_val_result, update_test_result


def train_one_epoch(
    args,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    model_ema: Optional[ModelEma] = None,
    print_freq: int = 100,
    logger=None,
    meas_force=True,
):
    """Train for one epoch.
    Keys in dataloader: ['z', 'pos', 'batch', 'y', 'dy']
    """

    model.train()
    criterion.train()

    loss_metrics = {"energy": AverageMeter(), "force": AverageMeter()}
    mae_metrics = {"energy": AverageMeter(), "force": AverageMeter()}

    start_time = time.perf_counter()

    task_mean = model.task_mean
    task_std = model.task_std

    max_steps = len(data_loader)
    for step, data in enumerate(data_loader):
        data = data.to(device)

        # energy, force
        pred_y, pred_dy, info = model(
            data=data, # for EquiformerV2
            node_atom=data.z,
            pos=data.pos,
            batch=data.batch,
            step=global_step,
            datasplit="train",
        )

        loss_e = criterion(pred_y, ((data.y - task_mean) / task_std))
        loss = args.energy_weight * loss_e
        if meas_force == True:
            loss_f = criterion(pred_dy, (data.dy / task_std))
            loss += args.force_weight * loss_f
        else:
            pred_dy, loss_f = get_force_placeholder(data.dy, loss_e)

        # If you use trajectory sampling, fp_correction automatically
        # aligns the tensors and applies your loss function.
        # loss_fn = lambda y_gt, y: ((y_gt - y) ** 2).mean()
        # train_loss = fp_correction(loss_fn, (y_train, y_pred))

        optimizer.zero_grad()
        loss.backward()
        # optionally clip and log grad norm
        if args.clip_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.clip_grad_norm,
            )
            # if args.global_step % args.log_every_step_minor == 0:
            wandb.log({"grad_norm": grad_norm}, step=global_step)
            # .log({"grad_norm": grad_norm}, step=self.step, split="train")
        else:
            # grad_norm = 0
            # for p in model.parameters():
            #     param_norm = p.grad.detach().data.norm(2)
            #     grad_norm += param_norm.item() ** 2
            # grad_norm = grad_norm ** 0.5
            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
            grad_norm = torch.cat(grads).norm()
            wandb.log({"grad_norm": grad_norm}, step=global_step)
        optimizer.step()

        loss_metrics["energy"].update(loss_e.item(), n=pred_y.shape[0])
        loss_metrics["force"].update(loss_f.item(), n=pred_dy.shape[0])

        energy_err = pred_y.detach() * task_std + task_mean - data.y
        energy_err = torch.mean(torch.abs(energy_err)).item()
        mae_metrics["energy"].update(energy_err, n=pred_y.shape[0])

        force_err = pred_dy.detach() * task_std - data.dy
        force_err = torch.mean(
            torch.abs(force_err)
        ).item()  # based on OC20 and TorchMD-Net, they average over x, y, z
        mae_metrics["force"].update(force_err, n=pred_dy.shape[0])

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        # logging
        if step % print_freq == 0 or step == max_steps - 1:
            w = time.perf_counter() - start_time
            e = (step + 1) / max_steps
            info_str = "Epoch: [{epoch}][{step}/{length}] \t".format(
                epoch=epoch, step=step, length=max_steps
            )
            info_str += "loss_e: {loss_e:.5f}, loss_f: {loss_f:.5f}, e_MAE: {e_mae:.5f}, f_MAE: {f_mae:.5f}, ".format(
                loss_e=loss_metrics["energy"].avg,
                loss_f=loss_metrics["force"].avg,
                e_mae=mae_metrics["energy"].avg,
                f_mae=mae_metrics["force"].avg,
            )
            info_str += "time/step={time_per_step:.0f}ms, ".format(
                time_per_step=(1e3 * w / e / max_steps)
            )
            info_str += "lr={:.2e}".format(optimizer.param_groups[0]["lr"])
            logger.info(info_str)

        if step % args.log_every_step_minor == 0:
            # log to wandb
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_e_mae": mae_metrics["energy"].avg,
                    "train_f_mae": mae_metrics["force"].avg,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

        global_step += 1

    return mae_metrics, loss_metrics, global_step


def evaluate(
    args,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    print_freq: int = 100,
    logger=None,
    print_progress=False,
    max_iter=-1,
    global_step=None,
    datasplit=None,
):

    if args.eval_mode is True:
        model.eval()
        criterion.eval()

    loss_metrics = {"energy": AverageMeter(), "force": AverageMeter()}
    mae_metrics = {"energy": AverageMeter(), "force": AverageMeter()}

    start_time = time.perf_counter()

    task_mean = model.task_mean
    task_std = model.task_std

    # with torch.no_grad(): # remove because of torchdeq

    # fixed-point reuse
    n_fsolver_steps = 0
    fixedpoint = None
    prev_idx = None
    max_steps = max_iter if max_iter != -1 else len(data_loader)
    for step, data in enumerate(data_loader):
        data = data.to(device)

        # if we pass step, things will be logged to wandb
        # note that global_step is only updated in train_one_epoch
        if step == max_steps - 1:
            pass_step = global_step
        else:
            pass_step = None

        # fixed-point reuse
        if datasplit == "test" and args.fpreuse_test == True:
            # assert that idx is consecutive
            if prev_idx is not None:
                assert torch.allclose(data.idx, prev_idx + 1)
            prev_idx = data.idx
            # call model and pass fixedpoint
            pred_y, pred_dy, fixedpoint, info = model(
                data=data, # for EquiformerV2
                node_atom=data.z,
                pos=data.pos,
                batch=data.batch,
                step=pass_step,
                datasplit=datasplit,
                return_fixedpoint=True,
                fixedpoint=fixedpoint,
            )
        else:
            # energy, force
            pred_y, pred_dy, info = model(
                data=data, # for EquiformerV2
                node_atom=data.z,
                pos=data.pos,
                batch=data.batch,
                step=pass_step,
                datasplit=datasplit,
                fixedpoint=None,
            )

        loss_e = criterion(pred_y, ((data.y - task_mean) / task_std))
        if args.meas_force == True:
            loss_f = criterion(pred_dy, (data.dy / task_std))
        else:
            pred_dy, loss_f = get_force_placeholder(data.dy, loss_e)

        optimizer.zero_grad()

        # --- metrics ---
        loss_metrics["energy"].update(loss_e.item(), n=pred_y.shape[0])
        loss_metrics["force"].update(loss_f.item(), n=pred_dy.shape[0])

        energy_err = pred_y.detach() * task_std + task_mean - data.y
        energy_err = torch.mean(torch.abs(energy_err)).item()
        mae_metrics["energy"].update(energy_err, n=pred_y.shape[0])
        force_err = pred_dy.detach() * task_std - data.dy
        force_err = torch.mean(
            torch.abs(force_err)
        ).item()  # based on OC20 and TorchMD-Net, they average over x, y, z
        mae_metrics["force"].update(force_err, n=pred_dy.shape[0])

        if 'nstep' in info:
            n_fsolver_steps += info['nstep'][0].mean().item()

        # --- logging ---
        if (step % print_freq == 0 or step == max_steps - 1) and print_progress:
            w = time.perf_counter() - start_time
            e = (step + 1) / max_steps
            info_str = "[{step}/{length}] \t".format(step=step, length=max_steps)
            info_str += "e_MAE: {e_mae:.5f}, f_MAE: {f_mae:.5f}, ".format(
                e_mae=mae_metrics["energy"].avg,
                f_mae=mae_metrics["force"].avg,
            )
            info_str += "time/step={time_per_step:.0f}ms".format(
                time_per_step=(1e3 * w / e / max_steps)
            )
            logger.info(info_str)

        if ((step + 1) >= max_steps):
            break
    
    eval_time = time.perf_counter() - start_time
    wandb.log({f"time_{datasplit}": eval_time}, step=global_step)

    if n_fsolver_steps > 0:
        n_fsolver_steps /= max_steps
        wandb.log({f"avg_n_fsolver_steps_{datasplit}": n_fsolver_steps}, step=global_step)

    return mae_metrics, loss_metrics


@hydra.main(config_name="md17", config_path="../equiformer/config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    from deq2ff.logging_utils import init_wandb

    init_wandb(args)

    # args: omegaconf.dictconfig.DictConfig -> dict
    # args = OmegaConf.to_container(args, resolve=True)

    main(args)


if __name__ == "__main__":
    hydra_wrapper()
