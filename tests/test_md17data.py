import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from pathlib import Path
from typing import Iterable, Optional

import sys, os

import equiformer.datasets.pyg.md17_backup as md17_dataset

from equiformer.logger import FileLogger

# import equiformer.nets
from equiformer.nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from equiformer.optim_factory import create_optimizer

from equiformer.engine import AverageMeter, compute_stats

import hydra
import wandb
import omegaconf
from omegaconf import DictConfig

ModelEma = ModelEmaV2

# silence:
# UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`.
# Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def run_test(args):

    # create output directory
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(f"args passed to {__file__} main():\n {args}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """ Dataset """
    train_dataset, val_dataset, test_dataset = md17_dataset.get_md17_datasets(
        root=os.path.join(args.data_path, args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
    )

    _log.info("")
    _log.info("Training set size:   {}".format(len(train_dataset)))
    _log.info("Validation set size: {}".format(len(val_dataset)))
    _log.info("Testing set size:    {}\n".format(len(test_dataset)))

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())

    # since dataset needs random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fixed point reuse
    shuffle = True
    if "fpreuse" in args and args.fpreuse is True:
        shuffle = False

    """ Data Loader """
    # When both batch_size and batch_sampler are None, automatic batching is disabled.
    # Each sample obtained from the dataset is processed with the function passed as collate_fn.
    # When automatic batching is disabled, the default collate_fn simply converts
    # NumPy arrays into PyTorch Tensors, and keeps everything else untouched.

    # if shuffle=False, each batch contains consecutive idx
    # -> important for fixed-point reuse
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        # enables faster data transfer to CUDA-enabled GPUs
        # Host to GPU copies are much faster when they originate from pinned (page-locked) memory
        pin_memory=args.pin_mem,
        # drops the last non-full batch
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    print(f"batch_size: {args.batch_size}")

    # train
    data_loader = train_loader
    for epoch in range(args.epochs):
        # z_star = None
        for step, data in enumerate(data_loader):
            data = data.to(device)

            # y: energy [batch_size, 1]
            # dy: forces [batch_size*num_atoms, 3]
            # pos: 3D positions of atoms [batch_size*num_atoms, 3]
            # z: atom type index, 1=H 6=C 8=O [batch_size*num_atoms]
            # ptr: pointer, e.g. (0, 21, 42, 63) for num_atoms=21 [batch_size+1]
            # batch: index to which batch an atom belongs, e.g. (0,0,...,1,...,2,...) [batch_size*num_atoms]

            # print("ptr:", data.ptr, type(data.ptr))
            print(f"batches: {data.batch}")
            print(f"z: {data.z}")

            exit()


@hydra.main(
    config_name="md17",
    config_path="../equiformer/config/equiformer",
    version_base="1.3",
)
def hydra_wrapper(args: DictConfig) -> None:
    from deq2ff.logging_utils import init_wandb

    args.wandb = False
    args.batch_size = 3
    # args.fpreuse = False

    init_wandb(args)
    run_test(args)


if __name__ == "__main__":
    # test
    hydra_wrapper()
