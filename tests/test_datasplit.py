import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from pathlib import Path
from typing import Iterable, Optional

import sys, os

# import equiformer.datasets.pyg.md17 as md17_dataset
import equiformer.datasets.pyg.md17revised as md17_dataset

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

split_file = '/ssd/gen/equilibrium-forcefields/datasets/md17/aspirin/splits.npz'
split = np.load(split_file)

print(split)

# train_idx = split['train_idx']
# valid_idx = split['valid_idx']
# test_idx = split['test_idx']


@hydra.main(config_name="md17", config_path="../equiformer/config/equiformer", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    # print(train_idx)torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """ Dataset """
    train_dataset, val_dataset, test_dataset = md17_dataset.get_rmd17_datasets(
        root=os.path.join(args.data_path, args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        return_idx=True,
        consecutive=True,
    )

    print(f'Train dataset: {train_dataset} {len(train_dataset)}')
    print(f'Val dataset: {val_dataset} {len(val_dataset)}')
    print(f'Test dataset: {test_dataset} {len(test_dataset)}')

if __name__ == '__main__':
    hydra_wrapper()