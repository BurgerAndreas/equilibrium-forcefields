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

import deq2ff
import deq2ff.logging_utils_deq as logging_utils_deq
from deq2ff.losses import (
    load_loss,
    L2MAELoss,
    contrastive_loss,
    TripletLoss,
    calc_triplet_loss,
)
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


fdir = "/ssd/gen/equilibrium-forcefields/models/md17/deq_equiformer_v2_oc20/ethanol/DEQE2FPCfpcrwpathdrop005targetethanol"
fname = "pathological_ep@95_e@nan_f@nan.pth.tar"


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
            test_patch_size=None,
            seed=args.seed,
            # order="consecutive_test" if args.fpreuse_test else None,
        )
    else:
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
        num_workers=args.num_workers,
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
        from deq2ff.data_utils import reorder_dataset

        test_dataset = reorder_dataset(test_dataset, args.eval_batch_size)
        _log.info(f"Reordered test dataset to be consecutive for fixed-point reuse.")
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=args.shuffle_test,
        drop_last=True,
    )

    """ Compute stats """
    # Compute _AVG_NUM_NODES, _AVG_DEGREE
    if args.compute_stats:
        avg_node, avg_edge, avg_degree = compute_stats(
            train_loader,
            max_radius=args.model.max_radius,
            filelog=_log,
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
            # args.checkpoint_path = os.path.join(args.output_dir, "checkpoint.pt.tar")
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
        node_embedding_batch_shape = shapes_to_log["NodeEmbeddingShape"]
        node_embedding_shape = list(node_embedding_batch_shape)
        node_embedding_shape[0] = node_embedding_shape[0] // args.batch_size
        print(f"node_embedding_shape: {node_embedding_shape}")
        print(f"node_embedding_batch_shape: {node_embedding_batch_shape}")
    except Exception as e:
        print(f"Failed to log shapes: {e}")
        node_embedding_batch_shape = None
        node_embedding_shape = None

    """ Training Loop """
    data_loader = train_loader
    collate = Collater(None, None)
    model.train()

    solver_kwargs = {}

    for step, data in enumerate(data_loader):
        data = data.to(device)

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

        break


@hydra.main(config_name="md17", config_path="../equiformer/config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    from deq2ff.logging_utils import init_wandb

    args.output_dir = None
    args.checkpoint_path = fdir + "/" + fname
    args.wandb = False

    init_wandb(args)

    # args: omegaconf.dictconfig.DictConfig -> dict
    # args = OmegaConf.to_container(args, resolve=True)

    # export PRINT_VALUES=1
    os.environ["PRINT_VALUES"] = "1"

    main(args)


if __name__ == "__main__":
    hydra_wrapper()
