import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from pathlib import Path
from typing import Iterable, Optional

import sys, os

import equiformer.datasets.pyg.md17 as md17_dataset
import equiformer.datasets.pyg.md17revised as rmd17_dataset

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

def test_old_is_new(args):

    """ Dataset """
    train_dataset, val_dataset, test_dataset = rmd17_dataset.get_rmd17_datasets(
        root=os.path.join(args.data_path, args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        return_idx=False,
        # order='consecutive',
    )

    print(f'Train dataset: {len(train_dataset)}')
    print(f'Val dataset: {len(val_dataset)}')
    print(f'Test dataset: {len(test_dataset)}')

    # Same as unrevised?
    train_dataset_old, _, _ = md17_dataset.get_md17_datasets(
        root=os.path.join(args.data_path, args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        return_idx=False,
        # order='consecutive',
    )

    for i in range(len(train_dataset)):
        assert torch.allclose(train_dataset[i].idx, train_dataset_old[i].idx), f'{i}'
        assert torch.allclose(train_dataset[i].y, train_dataset_old[i].y), f'{i}'
        assert torch.allclose(train_dataset[i].dy, train_dataset_old[i].dy), f'{i}'
        assert torch.allclose(train_dataset[i].pos, train_dataset_old[i].pos), f'{i}'
    
    print('All good!')
    return True

from deq2ff.data_utils import reorder_dataset

def test_consecutive_order(args):
    """ Dataset """
    train_dataset, val_dataset, test_dataset = rmd17_dataset.get_rmd17_datasets(
        root=os.path.join(args.data_path, args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        return_idx=False,
        order='consecutive',
    )

    print(f'Train dataset: {len(train_dataset)}')
    print(f'Val dataset: {len(val_dataset)}')
    print(f'Test dataset: {len(test_dataset)}')

    # Batches consecutive?
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True)

    test_dataset = reorder_dataset(test_dataset, args.eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True)


    print('')
    last_idx = None
    for step, data in enumerate(test_loader):
        
        # print(data)
        # print(f'idx: {data.idx}')

        if last_idx is not None:
            assert torch.allclose(data.idx, last_idx + 1), f'{step}'
        last_idx = data.idx
        
        if step >= 10:
            break
    
    print('All good!')
    return True

@hydra.main(config_name="md17", config_path="../equiformer/config/equiformer", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    # print(train_idx)torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
   
    test_old_is_new(args)

    test_consecutive_order(args)



if __name__ == '__main__':
    hydra_wrapper()