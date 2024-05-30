import argparse
import datetime
import itertools
import pickle
import subprocess
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.loader.dataloader import Collater

# from torch_geometric.data import collate

import os
import sys
import yaml

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
from equiformer.optim_factory import create_optimizer, scale_batchsize_lr

from equiformer.engine import AverageMeter, compute_stats

import torch
import torch.nn as nn
from torch_cluster import radius_graph
from torch_scatter import scatter

from torch.profiler import profile, record_function, ProfilerActivity

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

from typing import List
import tracemalloc

from equiformer_v2.oc20.trainer.base_trainer_oc20 import Normalizer

"""
Adapted from equiformer/main_md17.py
"""

import deq2ff
import deq2ff.logging_utils_deq as logging_utils_deq
from deq2ff.losses import (
    load_loss,
    L2MAELoss,
    contrastive_loss,
    TripletLoss,
    calc_triplet_loss,
)
from deq2ff.data_utils import reorder_dataset
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

file_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(file_dir)

# silence:
# UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`.
# Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
# torch_geometric/data/collate.py:150: UserWarning: An output with one or more elements was resized since it had shape, which does not match the required output shape
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_next_batch(dataset, batch, collate):
    """Get batch where indices are consecutive to previous batch."""
    # get the next timesteps
    idx = batch.idx + 1
    idx = idx.to("cpu")

    # only works when dataset is consecutive
    # make sure we don't go out of bounds
    len_train = len(dataset)
    idx = torch.where(idx >= len_train, idx - 2, idx)
    # idx.clamp_(0, len_train - 1)
    idx = idx.tolist()

    # If you want to access specific elements we need to use torch.utils.data.Dataset
    # DataLoader has no __getitem__ method (see in the source code for yourself).
    # DataLoader is used for iterating, not random access, over data (or batches of data).
    # index the dataset:                      <class 'equiformer.datasets.pyg.md_all.MDAll'>
    # with collate / next(iter(data_loader)): <class 'torch_geometric.data.batch.DataBatch'>
    next_data = collate([dataset[_idx] for _idx in idx])
    next_data = next_data.to(batch.idx.device)

    # assert idx and next_data.idx are (at max, if we use clamp) one apart
    assert (
        next_data.idx - batch.idx
    ).abs().float().mean() <= 1.0, f"idx: {idx}, next_data.idx: {next_data.idx}"

    return next_data


# def save_checkpoint():
#     torch.save(
#         {
#             "state_dict": model.state_dict(),
#             "optimizer": optimizer.state_dict(),
#             "lr_scheduler": lr_scheduler.state_dict(),
#             "epoch": epoch,
#             "global_step": global_step,
#             "best_metrics": best_metrics,
#             "best_ema_metrics": best_ema_metrics,
#         },
#         os.path.join(
#             args.output_dir,
#             "final_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
#                 epoch, test_err["energy"].avg, test_err["force"].avg
#             ),
#         ),
#     )


def remove_extra_checkpoints(output_dir, max_checkpoints, startswith="epochs@"):
    # list all files starting with "epochs@" in the output directory
    # and sort them by the epoch number
    # then remove all but the last 5 checkpoints
    checkpoints = sorted(
        [
            f
            for f in os.listdir(output_dir)
            if f.startswith(startswith) and f.endswith(".pth.tar")
        ],
        key=lambda x: int(x.split("@")[1].split("_")[0]),
    )
    for f in checkpoints[:-max_checkpoints]:
        os.remove(os.path.join(output_dir, f))


def get_force_placeholder(dy, loss_e):
    """if meas_force is False, return a placeholder for force prediction and loss_f"""
    # pred_dy = torch.zeros_like(data.dy)
    pred_dy = torch.full_like(dy, float("nan"))
    # loss_f = torch.zeros_like(loss_e)
    loss_f = torch.full_like(loss_e, float("nan"))
    return pred_dy, loss_f

def compute_loss(args, y, dy, target_y, target_dy, criterion_energy, criterion_force):
    """Fix output shapes and compute loss."""
    # reshape model output [B] (OC20) -> [B,1] (MD17)
    if args.unsqueeze_e_dim and y.dim() == 1:
        y = y.unsqueeze(-1)

    # reshape data [B,1] (MD17) -> [B] (OC20)
    if args.squeeze_e_dim and target_y.dim() == 2:
        target_y = target_y.squeeze(1)

    loss_e = criterion_energy(y, target_y)
    loss = args.energy_weight * loss_e
    if args.meas_force == True:
        loss_f = criterion_force(dy, target_dy)
        loss += args.force_weight * loss_f
    else:
        dy, loss_f = get_force_placeholder(target_dy, loss_e)
    return loss, loss_e, loss_f


def main(args):

    # create output directory
    if args.output_dir == "auto":
        # args.output_dir = os.path.join('outputs', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        # models/md17/equiformer/test
        mname = args.checkpoint_wandb_name  # wandb.run.name
        # remove special characters
        mname = "".join(e for e in mname if e.isalnum())
        args.output_dir = f"models/{args.dname}/{args.model.name}/{args.target}/{mname}"
        print(f"Set output directory automatically: {args.output_dir}")
    elif args.output_dir == "checkpoint_path":
        # args.output_dir = args.checkpoint_path
        args.output_dir = os.path.dirname(args.checkpoint_path)
        print(f"Set output directory: {args.output_dir}")
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    wandb.run.config.update({"output_dir": args.output_dir}, allow_val_change=True)

    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    # _log.info(
    #     f"Args passed to {__file__} main():\n {omegaconf.OmegaConf.to_yaml(args)}"
    # )

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
        test_dataset_full = test_dataset
    else:
        import equiformer.datasets.pyg.md_all as md_all

        train_dataset, val_dataset, test_dataset, test_dataset_full, all_dataset = md_all.get_md_datasets(
            root=args.data_path,
            dataset_arg=args.target,
            dname=args.dname,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=None,
            test_size_select=args.test_size, # doesn't influence data splitting
            seed=args.seed,
            order=md_all.get_order(args),
        )
        # assert that dataset is consecutive
        samples = Collater(follow_batch=None, exclude_keys=None)(
            [all_dataset[i] for i in range(10)]
        )
        assert torch.allclose(
            samples.idx, torch.arange(10)
        ), f"idx are not consecutive: {samples.idx}"

    _log.info("")
    _log.info("Training set size:   {}".format(len(train_dataset)))
    _log.info("Validation set size: {}".format(len(val_dataset)))
    _log.info("Testing set size:    {}".format(len(test_dataset)))

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    task_mean = float(y.mean())
    task_std = float(y.std())
    _log.info("Training set mean: {}, std: {}\n".format(task_mean, task_std))

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

    """ Data Loader """
    _log.info("Loading data...")
    # We don't need to shuffle because either the indices are already randomized
    # or we want to keep the order
    # we just keep the shuffle option for the sake of consistency with equiformer
    shuffle = True
    # if args.datasplit in ["fpreuse_ordered"]:
    #     shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    # idx are from the dataset e.g. (1, ..., 100k)
    # indices are from the DataLoader e.g. (0, ..., 1k)
    idx_to_indices = {
        idx.item(): i for i, idx in enumerate(train_loader.dataset.indices)
    }
    indices_to_idx = {v: k for k, v in idx_to_indices.items()}
    # added drop_last=True to avoid error with fixed-point reuse
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True
    )
    if args.datasplit.startswith("fpreuse"):
        # reorder test dataset to be consecutive
        test_dataset = reorder_dataset(test_dataset, args.eval_batch_size)
        _log.info(f"Reordered test dataset to be consecutive for fixed-point reuse.")
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=args.shuffle_test, drop_last=True
    )
    # full dataset for final evaluation
    test_loader_full = DataLoader(
        test_dataset_full, batch_size=args.eval_batch_size, shuffle=args.shuffle_test, drop_last=True
    )

    """ Compute stats """
    # Compute _AVG_NUM_NODES, _AVG_DEGREE
    if args.compute_stats:
        avg_node, avg_edge, avg_degree = compute_stats(
            train_loader,
            max_radius=args.model.max_radius,
            logger=_log,
            print_freq=args.print_freq,
        )
        print(
            f"\nComputed stats: \n\tavg_node={avg_node} \n\tavg_edge={avg_edge} \n\tavg_degree={avg_degree}\n"
        )
        return {
            "avg_node": avg_node,
            "avg_edge": avg_edge,
            "avg_degree": avg_degree.item(),
        }

    """ Instantiate Model """
    _log.info("Creating model...")
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
    _log.info(
        f"\nModel {args.model.name} created with kwargs:\n{omegaconf.OmegaConf.to_yaml(args.model)}"
    )
    # _log.info(f"Model: \n{model}")

    # log available memory
    if torch.cuda.is_available():
        _log.info(
            f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
    else:
        _log.info(f"Warning: torch.cuda not available!\n")

    # If you need to move a model to GPU via .cuda() , please do so before constructing optimizers for it.
    # Parameters of a model after .cuda() will be different objects with those before the call.
    model = model.to(device)

    """ Instantiate everything else """
    _log.info("Creating optimizer & co...")
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
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
    start_epoch = 0
    global_step = 0

    # criterion = L2MAELoss()
    loss_fn = load_loss({"energy": args.loss_energy, "force": args.loss_force})
    criterion_energy = loss_fn["energy"]
    criterion_force = loss_fn["force"]

    """ Load checkpoint """
    loaded_checkpoint = False
    if args.checkpoint_path is not None:
        if args.checkpoint_path == "auto":
            # args.checkpoint_path = os.path.join(args.output_dir, "checkpoint.pth.tar")
            args.checkpoint_path = args.output_dir
            _log.info(f"Auto checkpoint path: {args.checkpoint_path}")
        try:
            # pass either a checkpoint or a directory containing checkpoints
            # models/md17/deq_equiformer_v2_oc20/aspirin/DEQE2/epochs@0_e@4.6855_f@20.8729.pth.tar
            # models/md17/deq_equiformer_v2_oc20/aspirin/DEQE2
            if os.path.isdir(args.checkpoint_path):
                # get the latest checkpoint
                checkpoints = sorted(
                    [
                        f
                        for f in os.listdir(args.checkpoint_path)
                        if f.endswith(".pth.tar")
                    ],
                    key=lambda x: int(x.split("@")[1].split("_")[0]),
                )
                args.checkpoint_path = os.path.join(
                    args.checkpoint_path, checkpoints[-1]
                )
            state_dict = torch.load(args.checkpoint_path)
            # write state_dict
            model.load_state_dict(state_dict["state_dict"])
            optimizer.load_state_dict(state_dict["optimizer"])
            lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
            start_epoch = state_dict["epoch"]
            global_step = state_dict["global_step"]
            best_metrics = state_dict["best_metrics"]
            best_ema_metrics = state_dict["best_ema_metrics"]
            # log
            _log.info(f"Loaded model from {args.checkpoint_path}")
            loaded_checkpoint = True
        except Exception as e:
            # probably checkpoint not found
            _log.info(f"Error loading checkpoint: {e}")
    wandb.log({"start_epoch": start_epoch, "epoch": start_epoch}, step=global_step)

    # if we want to run inference only we want to make sure that the model is loaded
    if args.assert_checkpoint:
        assert (
            loaded_checkpoint
        ), f"Failed to load checkpoint at path={args.checkpoint_path}."
        assert (
            start_epoch >= args.epochs * 0.98
        ), f"Loaded checkpoint at path={args.checkpoint_path} isn't finished yet. start_epoch={start_epoch}."

    # watch gradients, weights, and activations
    # https://docs.wandb.ai/ref/python/watch
    if args.watch_model:
        wandb.watch(model, log="all", log_freq=args.log_every_step_major)

    # TODO
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )

    """ Log number of parameters """
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
        n_parameters = sum(
            p.numel() for p in model.energy_block.parameters() if p.requires_grad
        )
        _log.info("Number of EnergyBlock params: {}".format(n_parameters))
        wandb.run.summary["EnergyBlock Parameters"] = n_parameters
        # force prediction
        n_parameters = sum(
            p.numel() for p in model.force_block.parameters() if p.requires_grad
        )
        _log.info("Number of ForceBlock params: {}".format(n_parameters))
        wandb.run.summary["ForceBlock Parameters"] = n_parameters
        # print(
        #     f"AttributeError: '{model.__class__.__name__}' object has no attribute 'final_block'"
        # )

    """ Load dataset stats """
    # Overwrite _AVG_NUM_NODES and _AVG_DEGREE with the dataset statistics
    if args.load_stats:
        import json

        if type(args.load_stats) == str:
            stats = json.load(open(args.load_stats))
        else:
            # default: datasets/statistics.json
            _fpath = os.path.join("datasets", "statistics.json")
            stats = json.load(open(_fpath))
        # load statistics of molecule or dataset
        if args.use_dataset_avg_stats:
            _stats = stats[args.dname]["_avg"][str(float(args.model.max_radius))]
        else:
            try:
                _stats = stats[args.dname][args.target][
                    str(float(args.model.max_radius))
                ]
            except:
                print(
                    f"Could not find statistics for {args.dname}, {args.target}, {args.model.max_radius}"
                )
                print(f"Only found: {yaml.dump(stats)}")
                _stats = stats[args.dname]["_avg"][str(float(args.model.max_radius))]
        _AVG_NUM_NODES, _AVG_DEGREE = _stats["avg_node"], _stats["avg_degree"]
        # overwrite model parameters
        model._AVG_NUM_NODES = float(_AVG_NUM_NODES)
        model._AVG_DEGREE = float(_AVG_DEGREE)
        # write to wandb
        # V1: add to summary
        # wandb.run.summary["_AVG_NUM_NODES"] = _AVG_NUM_NODES
        # wandb.run.summary["_AVG_DEGREE"] = _AVG_DEGREE
        # V2: dont: this will overwrite the config
        # wandb.run.config.update({"model": {"_AVG_NUM_NODES": _AVG_NUM_NODES, "_AVG_DEGREE": _AVG_DEGREE}})
        # V3: this will add new entries to the config
        wandb.run.config.update(
            {"_AVG_NUM_NODES": _AVG_NUM_NODES, "_AVG_DEGREE": _AVG_DEGREE}
        )
        # V4: update all config args
        args.model._AVG_NUM_NODES = float(_AVG_NUM_NODES)
        args.model._AVG_DEGREE = float(_AVG_DEGREE)
        # wandb.config.update(args)
        _log.info(
            f"Loaded computed stats: _AVG_NUM_NODES={_AVG_NUM_NODES}, _AVG_DEGREE={_AVG_DEGREE}"
        )

    # update all config args
    # wandb.config.update(OmegaConf.to_container(args, resolve=True), allow_val_change=True)

    """ Dryrun of forward pass for testing """
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

    # print(f'pred_y: {pred_y.shape}')
    # print(f'pred_dy: {pred_dy.shape}')
    # print(f'data.y: {data.y.shape}')
    # print(f'data.dy: {data.dy.shape}')

    if args.test_forward:
        return True

    """ Dryrun for logging shapes """
    try:
        model.train()
        # criterion.train()
        criterion_energy.train()
        criterion_force.train()

        for step, data in enumerate(train_loader):
            data = data.to(device)

            # energy, force
            shapes_to_log = model.get_shapes(
                data=data,
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
        print(f"Shapes (target={args.target}):")
        ppr.pprint(shapes_to_log)
        wandb.run.summary.update(shapes_to_log)

        # TODO: might not work with input injection as concat
        NODE_EMBEDDING_BATCH_SHAPE = shapes_to_log["NodeEmbeddingShape"]
        NODE_EMBEDDING_SHAPE = list(NODE_EMBEDDING_BATCH_SHAPE)
        NODE_EMBEDDING_SHAPE[0] = NODE_EMBEDDING_SHAPE[0] // args.batch_size
        print(f"NODE_EMBEDDING_SHAPE: {NODE_EMBEDDING_SHAPE}")
        print(f"NODE_EMBEDDING_BATCH_SHAPE: {NODE_EMBEDDING_BATCH_SHAPE}")
    except Exception as e:
        print(f"Failed to log shapes: {e}")
        NODE_EMBEDDING_BATCH_SHAPE = None
        NODE_EMBEDDING_SHAPE = None

    """ Log memory usage """
    # Start recording memory snapshot history, initialized with a buffer
    # capacity of 100,000 memory events, via the `max_entries` field.
    if args.torch_record_memory:
        try:
            torch.cuda.memory._record_memory_history(
                max_entries=args.MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
            )
        except Exception as e:
            _log.info(f"Failed to record memory history {e}")

    """ Inference! """
    if args.evaluate:
        test_err, test_loss = evaluate(
            args=args,
            model=model,
            criterion_energy=criterion_energy,
            criterion_force=criterion_force,
            data_loader=test_loader,
            optimizer=optimizer,
            device=device,
            print_freq=args.print_freq,
            logger=_log,
            print_progress=True,
            max_iter=args.test_max_iter,
            global_step=global_step,
            datasplit="test",
            normalizers=normalizers,
        )
        wandb.log(
            {
                "test_e_mae": test_err["energy"].avg,
                "test_f_mae": test_err["force"].avg,
                # also save as best metrics?
                # "best_test_e_mae": test_err["energy"].avg,
                # "best_test_f_mae": test_err["force"].avg,
                # "epoch": start_epoch,
            },
            step=global_step,
        )
        if args.torch_record_memory:
            # Snapshots will save last `max_entries` number of memory events
            try:
                torch.cuda.memory._dump_snapshot(
                    f"{args.output_dir}/cuda_memory_snapshot_inference_s{global_step}.pickle"
                )
            except Exception as e:
                _log.info(f"Failed to capture memory snapshot {e}")
            # Stop recording memory snapshot history.
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except Exception as e:
                _log.info(f"Failed to stop recording memory history {e}")
        return True

    """ Train! """
    if NODE_EMBEDDING_SHAPE is not None:
        # empty list to store fixed-points across epochs
        # fixed_points = [None] * args.train_size
        fpdevice = device if args.fp_on_gpu else torch.device("cpu")
        # fixed_points = [torch.zeros(NODE_EMBEDDING_SHAPE, device=fpdevice)] * args.train_size
        fixed_points = torch.zeros(
            args.train_size, *NODE_EMBEDDING_SHAPE, device=fpdevice, requires_grad=False
        )
        # empty tensor to store fixed-points across epochs
        # fixed_points = torch.zeros(args.train_size, 3, device=device)
    else:
        fixed_points = None
        fpdevice = None
    _log.info("\nStart training!\n")
    start_time = time.perf_counter()
    final_epoch = 0
    for epoch in range(start_epoch, args.max_epochs):

        epoch_start_time = time.perf_counter()

        lr_scheduler.step(epoch)
        # print('lr:', optimizer.param_groups[0]["lr"])
        # print('lr:', lr_scheduler.get_last_lr())

        train_err, train_loss, global_step, fixed_points = train_one_epoch(
            args=args,
            model=model,
            # criterion=criterion,
            criterion_energy=criterion_energy,
            criterion_force=criterion_force,
            data_loader=train_loader,
            # all_loader=all_loader,
            all_dataset=all_dataset,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            model_ema=model_ema,
            print_freq=args.print_freq,
            logger=_log,
            normalizers=normalizers,
            fixed_points=fixed_points,
            indices_to_idx=indices_to_idx,
            idx_to_indices=idx_to_indices,
            fpdevice=fpdevice,
        )
        epoch_train_time = time.perf_counter() - epoch_start_time

        if args.torch_record_memory:
            # Snapshots will save last `max_entries` number of memory events
            try:
                torch.cuda.memory._dump_snapshot(
                    f"{args.output_dir}/cuda_memory_snapshot_e{epoch}_s{global_step}.pickle"
                )
            except Exception as e:
                _log.info(f"Failed to capture memory snapshot {e}")

        val_err, val_loss = evaluate(
            args=args,
            model=model,
            # criterion=criterion,
            criterion_energy=criterion_energy,
            criterion_force=criterion_force,
            data_loader=val_loader,
            optimizer=optimizer,
            device=device,
            print_freq=args.print_freq,
            logger=_log,
            print_progress=False,
            max_iter=-1,
            global_step=global_step,
            datasplit="val",
            normalizers=normalizers,
        )

        if (epoch + 1) % args.test_interval == 0:
            _log.info(f"Testing model after epoch {epoch+1}.")
            test_err, test_loss = evaluate(
                args=args,
                model=model,
                # criterion=criterion,
                criterion_energy=criterion_energy,
                criterion_force=criterion_force,
                data_loader=test_loader,
                optimizer=optimizer,
                device=device,
                print_freq=args.print_freq,
                logger=_log,
                print_progress=True,
                max_iter=args.test_max_iter,
                global_step=global_step,
                datasplit="test",
                normalizers=normalizers,
            )
        else:
            test_err, test_loss = None, None

        update_val_result, update_test_result = update_best_results(
            args, best_metrics, val_err, test_err, epoch
        )
        saved_best_checkpoint = False
        if update_val_result and args.save_best_val_checkpoint:
            _log.info(f"Saving best val checkpoint.")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_metrics": best_metrics,
                    "best_ema_metrics": best_ema_metrics,
                },
                os.path.join(
                    args.output_dir,
                    "best_val_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, val_err["energy"].avg, val_err["force"].avg
                    ),
                ),
            )
            remove_extra_checkpoints(
                args.output_dir, args.max_checkpoints, startswith="best_val_epochs@"
            )
            saved_best_checkpoint = True

        if update_test_result and args.save_best_test_checkpoint:
            _log.info(f"Saving best test checkpoint.")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_metrics": best_metrics,
                    "best_ema_metrics": best_ema_metrics,
                },
                os.path.join(
                    args.output_dir,
                    "best_test_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, test_err["energy"].avg, test_err["force"].avg
                    ),
                ),
            )
            remove_extra_checkpoints(
                args.output_dir, args.max_checkpoints, startswith="best_test_epochs@"
            )
            saved_best_checkpoint = True

        if (
            (epoch + 1) % args.test_interval == 0
            # and (not update_val_result)
            # and (not update_test_result)
            and not saved_best_checkpoint
            and args.save_checkpoint_after_test
        ):
            _log.info(f"Saving checkpoint.")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_metrics": best_metrics,
                    "best_ema_metrics": best_ema_metrics,
                },
                os.path.join(
                    args.output_dir,
                    "epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, test_err["energy"].avg, test_err["force"].avg
                    ),
                ),
            )
            remove_extra_checkpoints(
                args.output_dir, args.max_checkpoints, startswith="epochs@"
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
            "train_loss_e": train_err["energy"].avg,
            "train_loss_f": train_err["force"].avg,
            # TODO: other losses
            "val_e_mae": val_err["energy"].avg,
            "val_f_mae": val_err["force"].avg,
            "lr": optimizer.param_groups[0]["lr"],
            # allows us to plot against epoch
            # in the custom plots, click edit and select a custom x-axis
            "epoch": epoch,
            "time_train": time.perf_counter() - start_time,
            "time_per_epoch": time.perf_counter() - epoch_start_time,
            "time_train_per_epoch": epoch_train_time,
        }
        # if test_err is not None: # inside evaluate
        #     logs["test_e_mae"] = test_err["energy"].avg
        #     logs["test_f_mae"] = test_err["force"].avg
        # if global_step % args.log_every_step_minor == 0:
        wandb.log(logs, step=global_step)

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
                # criterion=criterion,
                criterion_energy=criterion_energy,
                criterion_force=criterion_force,
                data_loader=val_loader,
                optimizer=optimizer,
                device=device,
                print_freq=args.print_freq,
                logger=_log,
                print_progress=False,
                global_step=global_step,
                datasplit="ema_val",
                normalizers=normalizers,
            )
            optimizer.zero_grad(set_to_none=True)

            if (epoch + 1) % args.test_interval == 0:
                _log.info(f"Testing EMA model at epoch {epoch}")
                ema_test_err, _ = evaluate(
                    args=args,
                    model=model_ema.module,
                    # criterion=criterion,
                    criterion_energy=criterion_energy,
                    criterion_force=criterion_force,
                    data_loader=test_loader,
                    optimizer=optimizer,
                    device=device,
                    print_freq=args.print_freq,
                    logger=_log,
                    print_progress=True,
                    max_iter=args.test_max_iter,
                    global_step=global_step,
                    datasplit="ema_test",
                    normalizers=normalizers,
                )
                optimizer.zero_grad(set_to_none=True)
            else:
                ema_test_err, ema_test_loss = None, None

            update_val_result, update_test_result = update_best_results(
                args, best_ema_metrics, ema_val_err, ema_test_err, epoch
            )

            saved_best_ema_checkpoint = False
            if update_val_result and args.save_best_val_checkpoint:
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
                remove_extra_checkpoints(
                    args.output_dir,
                    args.max_checkpoints,
                    startswith="best_ema_val_epochs@",
                )
                saved_best_ema_checkpoint = True

            if update_test_result and args.save_best_test_checkpoint:
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
                remove_extra_checkpoints(
                    args.output_dir,
                    args.max_checkpoints,
                    startswith="best_ema_test_epochs@",
                )
                saved_best_ema_checkpoint = True

            if (
                (epoch + 1) % args.test_interval == 0
                # and (not update_val_result)
                # and (not update_test_result)
                and not saved_best_ema_checkpoint
                and args.save_checkpoint_after_test
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
                remove_extra_checkpoints(
                    args.output_dir, args.max_checkpoints, startswith="ema_epochs@"
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

        final_epoch = epoch
        # epoch done

    _log.info("\nAll epochs done!\nFinal test:")

    if args.torch_record_memory:
        # Stop recording memory snapshot history.
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception as e:
            _log.info(f"Failed to stop recording memory history {e}")

    # all epochs done
    # evaluate on the whole testing set
    if args.do_final_test:
        optimizer.zero_grad(set_to_none=True)
        test_err, test_loss = evaluate(
            args=args,
            model=model,
            # criterion=criterion,
            criterion_energy=criterion_energy,
            criterion_force=criterion_force,
            data_loader=test_loader_full, # test_loader
            optimizer=optimizer,
            device=device,
            print_freq=args.print_freq,
            logger=_log,
            print_progress=True,
            max_iter=args.test_max_iter_final,  # -1 means evaluate the whole dataset
            global_step=global_step,
            datasplit="test_final",
            normalizers=normalizers,
        )
        optimizer.zero_grad(set_to_none=True)

    # save the final model
    if args.save_final_checkpoint:
        _log.info(f"Saving final checkpoint")
        torch.save(
            {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": final_epoch,
                "global_step": global_step,
                "best_metrics": best_metrics,
                "best_ema_metrics": best_ema_metrics,
            },
            os.path.join(
                args.output_dir,
                "final_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                    final_epoch, test_err["energy"].avg, test_err["force"].avg
                ),
            ),
        )

    _log.info(
        f"Final test error: MAE_e={test_err['energy'].avg}, MAE_f={test_err['force'].avg}"
    )
    # log to wandb
    wandb.log(
        {
            "final_test_e_mae": test_err["energy"].avg,
            "final_test_f_mae": test_err["force"].avg,
        },
        step=global_step,
    )
    _log.info(f"Done!")
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
    # criterion: torch.nn.Module,
    criterion_energy: torch.nn.Module,
    criterion_force: torch.nn.Module,
    data_loader: Iterable,
    # all_loader: Iterable,
    all_dataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    model_ema: Optional[ModelEma] = None,
    print_freq: int = 100,
    logger=None,
    normalizers={"energy": None, "force": None},
    # fixed-point reuse
    fixed_points=None,
    indices_to_idx=None,
    idx_to_indices=None,
    fpdevice=None,
):
    """Train for one epoch.
    Keys in dataloader: ['z', 'pos', 'batch', 'y', 'dy']
    """
    collate = Collater(None, None)

    model.train()
    criterion_energy.train()
    criterion_force.train()
    # crit_fpc = lambda x, y: (x - y).abs().mean()
    criterion_fpc = nn.MSELoss()
    criterion_contrastive = nn.MSELoss()
    criterion_fpr = nn.MSELoss()

    loss_metrics = {"energy": AverageMeter(), "force": AverageMeter()}
    mae_metrics = {"energy": AverageMeter(), "force": AverageMeter()}

    start_time = time.perf_counter()

    task_mean = model.task_mean
    task_std = model.task_std

    # triplet loss
    triplet_lossfn = TripletLoss(margin=args.tripletloss_margin)

    optimizer.zero_grad(set_to_none=True)

    # statistics over epoch
    abs_fixed_point_error = []
    rel_fixed_point_error = []
    f_steps_to_fixed_point = []

    # broyden solver outpus NaNs if it diverges
    # count the number of NaNs and stop training if it exceeds a threshold
    isnan_cnt = 0

    # for debugging
    # if we don't set model.eval()
    # params like the batchnorm layers in the model will be updated.
    # Even without optimizing parameters, validation outputs will change after a forward pass.

    if args.torch_profile:
        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
            activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True
        )
        prof.start()


    max_steps = len(data_loader)
    for rep in range(args.epochs_per_epochs):
        for step, data in enumerate(data_loader):
            data = data.to(device)

            # for sparse fixed-point correction loss
            solver_kwargs = {}
            if args.fpc_freq > 0:
                if args.fpc_rand:
                    # randomly uniform sample indices
                    solver_kwargs["indexing"] = torch.randperm(range(model.deq.f_max_iter))[
                        : args.fpc_freq
                    ]
                else:
                    # uniformly spaced indices
                    solver_kwargs["n_states"] = args.fpc_freq

            if args.fpreuse_across_epochs and epoch >= args.fpreuse_start_epoch:
                indices = [idx_to_indices[_idx.item()] for _idx in data.idx]
                # print(f'idx: {[_idx.item() for _idx in data.idx]}, indices: {indices}')
                # get previous fixed points via index
                # _fp_prev = fixed_points[step].to(device)
                # _fp_prev = fixed_points[indices].to(device)
                _fp_prev = torch.cat([fixed_points[_idx] for _idx in indices], dim=0)
                # _fp_prev = collate([fixed_points[_idx] for _idx in indices])
                # energy, force
                pred_y, pred_dy, fp, info = model(
                    data=data,  # for EquiformerV2
                    node_atom=data.z,
                    pos=data.pos,
                    batch=data.batch,
                    step=global_step,
                    datasplit="train",
                    solver_kwargs=solver_kwargs,
                    fpr_loss=args.fpr_loss,
                    # fixed-point reuse
                    return_fixedpoint=True,
                    fixedpoint=_fp_prev,
                )
                assert (
                    _fp_prev.shape == fp.shape
                ), f"Fixed-point shape mismatch: {_fp_prev.shape} != {fp.shape}"
                # split up fixed points [B*N, D, C] -> [B, N, D, C]
                # [84, 16, 64] -> [4, 21, 16, 64]
                fp = fp.view(args.batch_size, -1, *fp.shape[1:])
                # store fixed points
                # fixed_points[indices] = fp.detach()
                for _idx, _fp in zip(indices, fp):
                    # TODO is clone necessary?
                    # fixed_points[_idx] = _fp.detach().clone().to(fpdevice)
                    fixed_points[_idx] = _fp.detach().to(fpdevice)
                # print(' nsteps:', info["nstep"][0].item())
            else:
                # energy, force
                pred_y, pred_dy, info = model(
                    data=data,  # for EquiformerV2
                    node_atom=data.z,
                    pos=data.pos,
                    batch=data.batch,
                    step=global_step,
                    datasplit="train",
                    solver_kwargs=solver_kwargs,
                    fpr_loss=args.fpr_loss,
                )

            target_y = normalizers["energy"](data.y)
            target_dy = normalizers["force"](data.dy)

            # reshape model output [B] (OC20) -> [B,1] (MD17)
            if args.unsqueeze_e_dim and pred_y.dim() == 1:
                pred_y = pred_y.unsqueeze(-1)

            # reshape data [B,1] (MD17) -> [B] (OC20)
            if args.squeeze_e_dim and target_y.dim() == 2:
                target_y = target_y.squeeze(1)

            loss_e = criterion_energy(pred_y, target_y)
            loss = args.energy_weight * loss_e
            if args.meas_force == True:
                loss_f = criterion_force(pred_dy, target_dy)
                loss += args.force_weight * loss_f
            else:
                pred_dy, loss_f = get_force_placeholder(data.dy, loss_e)

            # Fixed-point correction loss
            # for superior performance and training stability
            # https://arxiv.org/abs/2204.08442
            if args.fpc_freq > 0:
                # I think this is never invoked if DEQSliced is used

                # If you use trajectory sampling, fp_correction automatically
                # aligns the tensors and applies your loss function.
                # loss_fn = lambda y_gt, y: ((y_gt - y) ** 2).mean()
                # train_loss = fp_correction(loss_fn, (y_train, y_pred))

                # do it manually instead
                if len(info["z_pred"]) > 1:
                    z_preds = info["z_pred"]
                    # last z is fixed point that is in the main loss
                    loss_fpc = 0
                    for z_pred in z_preds[:-1]:
                        _y, _dy, _ = model.decode(
                            data = data,
                            z = z_pred,
                            info={},
                        )

                        _loss_fpc, _, _ = compute_loss(
                            args=args,
                            y=_y,
                            dy=_dy,
                            target_y=target_y,
                            target_dy=target_dy,
                            criterion_energy=criterion_energy,
                            criterion_force=criterion_force,
                        )
                        loss_fpc += _loss_fpc

                    # reweight the loss
                    loss_fpc *= args.fpc_weight
                    loss += loss_fpc
                    if wandb.run is not None:
                        wandb.log(
                            {"fpc_loss_scaled": loss_fpc.item()},
                            step=global_step,
                        )

            # Contrastive loss
            if args.contrastive_loss not in [False, None]:
                # DEPRECATED: consecutive dataset won't converge, irrespective of loss
                if args.contrastive_loss.endswith("ordered"):
                    assert (
                        data.idx[0] - data.idx[1]
                    ).abs() == 1, (
                        f"Contrastive loss requires consecutive indices {data.idx}"
                    )
                    if args.contrastive_loss.startswith("triplet"):
                        closs = calc_triplet_loss(
                            info["z_pred"][-1], data, triplet_lossfn
                        )
                    elif args.contrastive_loss.startswith("next"):
                        assert (
                            data.idx[0] - data.idx[1]
                        ).abs() == 1, (
                            f"Contrastive loss requires consecutive indices {data.idx}"
                        )
                        closs = contrastive_loss(
                            info["z_pred"][-1],
                            data,
                            closs_type=args.contrastive_loss,
                            squared=True,
                        )

                elif args.contrastive_loss == "next":
                    next_data = get_next_batch(
                        dataset=all_dataset, batch=data, collate=collate
                    )
                    next_data = next_data.to(device)

                    # get correct fixed-point of next timestep
                    with torch.set_grad_enabled(args.contrastive_w_grad):
                        next_pred_y, next_pred_dy, next_info = model(
                            data=next_data,  # for EquiformerV2
                            node_atom=next_data.z,
                            pos=next_data.pos,
                            batch=next_data.batch,
                            step=None,
                            datasplit=None,
                        )
                    # loss
                    z_next_true = next_info["z_pred"][-1]
                    closs = criterion_contrastive(z_next_true, info["z_pred"][-1])

                else:
                    raise NotImplementedError(
                        f"Contrastive loss {args.contrastive_loss} not implemented."
                    )

                closs = args.contrastive_weight * closs
                loss += closs
                if wandb.run is not None:
                    wandb.log(
                        {"contrastive_loss_scaled": closs.item()}, step=global_step
                    )

            if args.fpr_loss == True:
                next_data = get_next_batch(
                    dataset=all_dataset, batch=data, collate=collate
                )
                next_data = next_data.to(device)
                # get correct fixed-point of next timestep
                with torch.set_grad_enabled(args.fpr_w_grad):
                    next_pred_y, next_pred_dy, next_info = model(
                        data=next_data,  # for EquiformerV2
                        node_atom=next_data.z,
                        pos=next_data.pos,
                        batch=next_data.batch,
                        step=None,
                        datasplit=None,
                    )
                # loss
                z_next_true = next_info["z_pred"][-1]
                fpr_loss = criterion_fpr(z_next_true, info["z_next"])
                loss += args.fpr_weight * fpr_loss
                wandb.log(
                    {"scaled_fpr_loss": (args.fpr_weight * fpr_loss).item()},
                    step=global_step,
                )


            if torch.isnan(pred_y).any():
                isnan_cnt += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # optionally clip and log grad norm
            if args.clip_grad_norm:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=args.clip_grad_norm,
                )
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

            # if args.lr > 0:
            optimizer.step()

            if len(info) > 0:
                # log fixed-point trajectory
                # if args.log_fixed_point_trace_train:
                logging_utils_deq.log_fixed_point_error(
                    info,
                    step=global_step,
                    datasplit="train",
                    log_trace_freq=args.log_trace_freq,
                )
                abs_fixed_point_error.append(info["abs_trace"].mean(dim=0)[-1].item())
                rel_fixed_point_error.append(info["rel_trace"].mean(dim=0)[-1].item())
                f_steps_to_fixed_point.append(info["nstep"].mean().item())

            loss_metrics["energy"].update(loss_e.item(), n=pred_y.shape[0])
            loss_metrics["force"].update(loss_f.item(), n=pred_dy.shape[0])
            # TODO: other losses

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
                logs = {
                    "train_loss": loss.item(),
                    "grad_norm": grad_norm.item(),
                    # energy
                    "energy_pred_mean": pred_y.mean().item(),
                    "energy_pred_std": pred_y.std().item(),
                    "energy_pred_min": pred_y.min().item(),
                    "energy_pred_max": pred_y.max().item(),
                    "energy_target_mean": target_y.mean().item(),
                    "energy_target_std": target_y.std().item(),
                    "energy_target_min": target_y.min().item(),
                    "energy_target_max": target_y.max().item(),
                    "scaled_energy_loss": (args.energy_weight * loss_e).item(),
                    # force
                    "force_pred_mean": pred_dy.mean().item(),
                    "force_pred_std": pred_dy.std().item(),
                    "force_pred_min": pred_dy.min().item(),
                    "force_pred_max": pred_dy.max().item(),
                    "force_target_mean": target_dy.mean().item(),
                    "force_target_std": target_dy.std().item(),
                    "force_target_min": target_dy.min().item(),
                    "force_target_max": target_dy.max().item(),
                    "scaled_force_loss": (args.force_weight * loss_f).item(),
                }
                if "z_pred" in info:
                    logs["len_z_pred"] = len(info["z_pred"])
                wandb.log(logs, step=global_step)

            global_step += 1

            if args.torch_profile:
                prof.step()

        # end of epoch
        if args.torch_profile:
            prof.stop()
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            # print location of the trace file
            mname = args.checkpoint_wandb_name  # wandb.run.name
            # remove special characters
            mname = "".join(e for e in mname if e.isalnum())
            prof.export_chrome_trace(f"{parent_dir}/traces/{mname}.json")
            print('Saved trace to:', f"{parent_dir}/traces/{mname}.json")
            exit()

        # log fixed-point statistics
        if len(abs_fixed_point_error) > 0:
            wandb.log(
                {
                    "abs_fixed_point_error_train": np.mean(abs_fixed_point_error),
                    "rel_fixed_point_error_train": np.mean(rel_fixed_point_error),
                    "f_steps_to_fixed_point_train": np.mean(f_steps_to_fixed_point),
                },
                step=global_step,
            )
            abs_fixed_point_error = []
            rel_fixed_point_error = []
            f_steps_to_fixed_point = []

        # if loss_metrics is all nan
        # probably because deq_kwargs.f_solver=broyden,anderson did not converge
        if isnan_cnt > max_steps // 2:
            # save checkpoint as pathological example
            _cname = "pathological_ep@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                epoch, mae_metrics["energy"].avg, mae_metrics["force"].avg
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    # "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    # "best_metrics": best_metrics,
                    # "best_ema_metrics": best_ema_metrics,
                },
                os.path.join(args.output_dir, _cname),
            )
            print(f"Saved pathological example at epoch {epoch} to {args.output_dir}/{_cname}")
            raise ValueError(
                f"Most energy predictions are nan ({isnan_cnt}/{max_steps}). Try deq_kwargs.f_solver=fixed_point_iter"
            )

    return mae_metrics, loss_metrics, global_step, fixed_points


def evaluate(
    args,
    model: torch.nn.Module,
    # criterion: torch.nn.Module,
    criterion_energy: torch.nn.Module,
    criterion_force: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    print_freq: int = 100,
    logger=None,
    print_progress=False,
    max_iter=-1,
    global_step=None,
    datasplit=None,
    normalizers={"energy": None, "force": None},
):
    """Val or test split."""

    # Techniques for faster inference
    # e.g. fixed-point reuse, relaxed FP-error threshold, etc.
    # for simplicity we only apply it to test (val is too small)
    if datasplit == "test" and "deq_kwargs_test" in args:
        solver_kwargs = args.deq_kwargs_test
    else:
        solver_kwargs = {}

    if args.test_w_eval_mode is True:
        model.eval()
        # criterion.eval()
        criterion_energy.eval()
        criterion_force.eval()
    
    loss_per_idx = False
    if args.eval_batch_size == 1:
        loss_per_idx = True

    task_mean = model.task_mean
    task_std = model.task_std

    # remove because of torchdeq and force prediction via dE/dx
    # with torch.no_grad():
    with torch.set_grad_enabled(args.test_w_grad):

        # if we use fpreuse_test, also try without to get a comparison
        if datasplit == "test" and args.fpreuse_test == True:
            if args.fpreuse_test_only == True:
                fpreuse_list = [True]
            else:
                # important that fpreuse comes first
                fpreuse_list = [True, False]
        else:
            fpreuse_list = [False]
        for fpreuse_test in fpreuse_list:
            # name for logging
            _datasplit = f"{datasplit}_fpreuse" if fpreuse_test else datasplit

            # initialize metrics
            loss_metrics = {"energy": AverageMeter(), "force": AverageMeter()}
            mae_metrics = {"energy": AverageMeter(), "force": AverageMeter()}
            abs_fixed_point_error = []
            rel_fixed_point_error = []
            f_steps_to_fixed_point = []
            model_forward_time = []
            n_fsolver_steps = []
            if loss_per_idx:
                # wandb table with columns: idx, e_mae, f_mae, nstep
                idx_table = wandb.Table(columns=["idx", "e_mae", "f_mae", "nstep"])

            fixedpoint = None
            prev_idx = None
            max_steps = max_iter if max_iter != -1 else len(data_loader)

            start_time = time.perf_counter()
            for step, data in enumerate(data_loader):
                data = data.to(device)

                # if we pass step, things will be logged to wandb
                # note that global_step is only updated in train_one_epoch
                # which is why we only want to log once per evaluation
                if step == max_steps - 1:
                    pass_step = global_step
                else:
                    pass_step = None

                forward_start_time = time.perf_counter()
                # fixed-point reuse
                if fpreuse_test == True:
                    # assert that idx is consecutive
                    if prev_idx is not None:
                        assert torch.allclose(data.idx, prev_idx + 1) or args.shuffle_test
                    prev_idx = data.idx
                    # call model and pass fixedpoint
                    pred_y, pred_dy, fixedpoint, info = model(
                        data=data,  # for EquiformerV2
                        node_atom=data.z,
                        pos=data.pos,
                        batch=data.batch,
                        step=pass_step,
                        datasplit=_datasplit,
                        return_fixedpoint=True,
                        fixedpoint=fixedpoint,
                        solver_kwargs=solver_kwargs,
                    )
                    # REMOVE
                    # print(f'step: {step}. idx: {data.idx}.') 
                    # print(f' pred_y: {pred_y.shape}, pred_dy: {pred_dy.shape}, fixedpoint: {fixedpoint.shape}')
                else:
                    # energy, force
                    pred_y, pred_dy, info = model(
                        data=data,  # for EquiformerV2
                        node_atom=data.z,
                        pos=data.pos,
                        batch=data.batch,
                        step=pass_step,
                        datasplit=datasplit,
                        fixedpoint=None,
                        solver_kwargs=solver_kwargs,
                    )
                model_forward_time += [time.perf_counter() - forward_start_time]

                target_y = normalizers["energy"](data.y)
                target_dy = normalizers["force"](data.dy)

                target_y = normalizers["energy"](data.y)
                target_dy = normalizers["force"](data.dy)

                # reshape model output [B] (OC20) -> [B,1] (MD17)
                if args.unsqueeze_e_dim and pred_y.dim() == 1:
                    pred_y = pred_y.unsqueeze(-1)

                # reshape data [B,1] (MD17) -> [B] (OC20)
                if args.squeeze_e_dim and target_y.dim() == 2:
                    target_y = pred_y.squeeze(1)

                loss_e = criterion_energy(pred_y, target_y)
                if args.meas_force == True:
                    loss_f = criterion_force(pred_dy, target_dy)
                else:
                    pred_dy, loss_f = get_force_placeholder(data.dy, loss_e)

                optimizer.zero_grad(set_to_none=True)

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
                
                # Remove
                # print(f' loss_e: {loss_e}, loss_f: {loss_f}')
                # print(f' enery_err: {energy_err}, force_err: {force_err}')

                # --- logging ---
                if len(info) > 0:
                    if pass_step is not None:
                        # log fixed-point trajectory once per evaluation
                        logging_utils_deq.log_fixed_point_error(
                            info,
                            step=global_step,
                            datasplit=_datasplit,
                        )
                    abs_fixed_point_error.append(
                        info["abs_trace"].mean(dim=0)[-1].item()
                    )
                    rel_fixed_point_error.append(
                        info["rel_trace"].mean(dim=0)[-1].item()
                    )
                    # duplicates kept for legacy reasons
                    f_steps_to_fixed_point.append(info["nstep"].mean().item())
                    n_fsolver_steps.append(info["nstep"].mean().item())

                if (step % print_freq == 0 or step == max_steps - 1) and print_progress:
                    w = time.perf_counter() - start_time
                    e = (step + 1) / max_steps
                    info_str = (
                        f"[{step}/{max_steps}]{'(fpreuse)' if fpreuse_test else ''} \t"
                    )
                    info_str += "e_MAE: {e_mae:.5f}, f_MAE: {f_mae:.5f}, ".format(
                        e_mae=mae_metrics["energy"].avg,
                        f_mae=mae_metrics["force"].avg,
                    )
                    info_str += "time/step={time_per_step:.0f}ms".format(
                        time_per_step=(1e3 * w / e / max_steps)
                    )
                    logger.info(info_str)
                
                if loss_per_idx:
                    # "idx", "e_mae", "f_mae", "nstep"
                    idx_table.add_data(data.idx.item(), energy_err, force_err, info["nstep"].mean().item())

                if (step + 1) >= max_steps:
                    break

            # test set finished
            eval_time = time.perf_counter() - start_time  # time for whole test set
            _logs = {
                f"{_datasplit}_e_mae": mae_metrics["energy"].avg,
                f"{_datasplit}_f_mae": mae_metrics["force"].avg,
                f"time_{_datasplit}": eval_time,
                f"time_forward_per_batch_{_datasplit}": np.mean(model_forward_time),
                # f"time_forward_per_batch_std_{_datasplit}": np.std(model_forward_time)
                f"time_forward_total_{_datasplit}": np.sum(model_forward_time),
            }
            # log the time
            wandb.log(
                {
                    f"time_{_datasplit}": eval_time,
                    f"time_forward_per_batch_{_datasplit}": np.mean(model_forward_time),
                    # f"time_forward_per_batch_std_{_datasplit}": np.std(model_forward_time),
                    f"time_forward_total_{_datasplit}": np.sum(model_forward_time),
                },
                step=global_step,
            )

            # log the table
            if loss_per_idx:
                wandb.log({f"idx_table_{_datasplit}": idx_table}, step=global_step)

            # log fixed-point statistics
            if len(abs_fixed_point_error) > 0:
                wandb.log(
                    {
                        f"abs_fixed_point_error_{_datasplit}": np.mean(
                            abs_fixed_point_error
                        ),
                        f"rel_fixed_point_error_{_datasplit}": np.mean(
                            rel_fixed_point_error
                        ),
                        f"f_steps_to_fixed_point_{_datasplit}": np.mean(
                            f_steps_to_fixed_point
                        ),
                    },
                    step=global_step,
                )
                _logs.update(
                    {
                        f"abs_fixed_point_error_{_datasplit}": np.mean(
                            abs_fixed_point_error
                        ),
                        f"rel_fixed_point_error_{_datasplit}": np.mean(
                            rel_fixed_point_error
                        ),
                        f"f_steps_to_fixed_point_{_datasplit}": np.mean(
                            f_steps_to_fixed_point
                        ),
                    }
                )

            if len(n_fsolver_steps) > 0:
                wandb.log(
                    {f"avg_n_fsolver_steps_{_datasplit}": np.mean(n_fsolver_steps)},
                    step=global_step,
                )
                # log the full list
                wandb.log(
                    {f"n_fsolver_steps_{_datasplit}": n_fsolver_steps}, step=global_step
                )

            # log test error
            wandb.log(
                {
                    f"{_datasplit}_e_mae": mae_metrics["energy"].avg,
                    f"{_datasplit}_f_mae": mae_metrics["force"].avg,
                },
                step=global_step,
            )

            print(f"Finished {_datasplit} evaluation.")
            for k, v in _logs.items():
                logger.info(f" {k}: {v}")

        # fp_reuse True/False finished

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
