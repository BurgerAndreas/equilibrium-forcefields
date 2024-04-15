import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from pathlib import Path
from typing import Iterable, Optional

import sys, os

import equiformer.datasets.pyg.md17 as md17_dataset
import equiformer.datasets.pyg.md_all as rmd17_dataset

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

import inspect

split_file = "/ssd/gen/equilibrium-forcefields/datasets/md17/aspirin/splits.npz"
split = np.load(split_file)

# print(split)

# train_idx = split['train_idx']
# valid_idx = split['valid_idx']
# test_idx = split['test_idx']


def test_revised_dataset_creation(args):
    """Test if the revised DatasetCreator can load the original dataset."""
    print("\n", "-" * 80, "\n", inspect.currentframe().f_code.co_name)

    """ Dataset """
    # new dataloader, old dataset
    train_dataset, val_dataset, test_dataset = rmd17_dataset.get_md_datasets(
        root=args.data_path,
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        return_idx=False,
    )

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Val dataset: {len(val_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")

    # old dataloader, old dataset
    train_dataset_old, _, _ = md17_dataset.get_md17_datasets(
        root=os.path.join(args.data_path, 'md17', args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        return_idx=False,
    )

    for i in range(len(train_dataset)):
        assert torch.allclose(train_dataset[i].idx, train_dataset_old[i].idx), f"{i}"
        assert torch.allclose(train_dataset[i].y, train_dataset_old[i].y), f"{i}"
        assert torch.allclose(train_dataset[i].dy, train_dataset_old[i].dy), f"{i}"
        assert torch.allclose(train_dataset[i].pos, train_dataset_old[i].pos), f"{i}"
        if i % 100 == 0:
            print(i)

    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())

    y_old = torch.cat([batch.y for batch in train_dataset_old], dim=0)

    assert torch.allclose(y, y_old), f"y"

    print(inspect.currentframe().f_code.co_name, "passed!")
    return True


def test_revisedold_equal_original(args):
    """Test if the `old` data in the revised dataset == original dataset."""
    print("\n", "-" * 80, "\n", inspect.currentframe().f_code.co_name)

    # If we don't specify the order, it will be a random permutation.
    # Since the length of the datasets is different,
    # the random permutation will be different.
    # original dataset: 211,762 samples
    # revised dataset: 100,000 samples
    order = "consecutive"
    # order = None
    # max_samples = 1e5 # 100k
    max_samples = -1

    # args.batch_size = 2
    # set_seed(args.seed)

    """ Dataset """
    # new dataloader, old data
    train, val, _ = rmd17_dataset.get_md_datasets(
        root=args.data_path,
        dataset_arg=args.target,
        train_size=0.5,
        val_size=0.5,
        test_size=None,
        max_samples=max_samples,
        seed=args.seed,
        revised=False,  # <--- Old data, old source
        order=order,
    )
    # get all samples
    dataset_og = torch.utils.data.ConcatDataset([train, val])
    print(f"Length original dataset: {len(dataset_og)}")

    # new dataloader, new dataset source, old data
    train, val, _ = rmd17_dataset.get_md_datasets(
        root=args.data_path,
        dataset_arg=args.target,
        train_size=0.5,  # get all samples
        val_size=0.5,
        test_size=None,
        seed=args.seed,
        max_samples=max_samples,
        revised=True,  # <--- Old data, new source
        revised_old=True,  # <--- Old data, new source
        order=order,
    )
    dataset_revisedold = torch.utils.data.ConcatDataset([train, val])
    print(f"Length revised_old dataset: {len(dataset_revisedold)}")

    a = dataset_og
    b = dataset_revisedold
    for i in range(len(b)):  # loop over revised dataset
        i_old = int(b[i].old_idx.item())
        # idx should be the same, because idx is the order in the dataset processing
        assert torch.allclose(a[i].idx, b[i].idx), f"{i}: {a[i].idx} != {b[i].idx}"
        # for everything else, we need to use the old_idx
        assert torch.allclose(a[i_old].y, b[i].y), f"{i}: {a[i].y} != {b[i].y}"
        assert torch.allclose(a[i_old].dy, b[i].dy), f"{i}: {a[i].dy} != {b[i].dy}"
        assert torch.allclose(a[i_old].pos, b[i].pos), f"{i}: {a[i].pos} != {b[i].pos}"
        if i % 10000 == 0:
            print(i, "check")

    # 211,762
    y = torch.cat([batch.y for batch in dataset_og], dim=0)
    mean = float(y.mean())
    std = float(y.std())

    # 100,000
    y_old = torch.cat([batch.y for batch in dataset_revisedold], dim=0)
    mean_old = float(y_old.mean())

    print(f"Difference in mean: {mean - mean_old}")

    print(inspect.currentframe().f_code.co_name, "passed!")
    return True


def test_load_revised_split(args):
    """Test loading the provided split of the revised dataset (for shape and dtype)."""
    print("\n", "-" * 80, "\n", inspect.currentframe().f_code.co_name)

    """ Dataset """
    train_dataset, val_dataset, test_dataset = rmd17_dataset.get_md_datasets(
        root=args.data_path,
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        revised=True,
        revised_old=False,
        load_splits=False,  # <--- Do not load the splits
        return_idx=True,
    )

    train_dataset_loaded, val_dataset, test_dataset = rmd17_dataset.get_md_datasets(
        root=args.data_path,
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        revised=True,
        revised_old=False,
        load_splits=True,  # <--- Load the splits
        return_idx=True,
    )

    # print('train_dataset', train_dataset.shape)
    # print('train_dataset_loaded', train_dataset_loaded.shape)

    # print('train_dataset', train_dataset)
    # print('train_dataset_loaded', train_dataset_loaded)

    # assert torch.equal(train_dataset.shape, train_dataset_loaded.shape), f'train_dataset'
    assert (
        train_dataset.shape == train_dataset_loaded.shape
    ), f"{train_dataset.shape} == {train_dataset_loaded.shape}"
    assert (
        train_dataset.dtype == train_dataset_loaded.dtype
    ), f"{train_dataset.dtype} == {train_dataset_loaded.dtype}"
    print(inspect.currentframe().f_code.co_name, "passed!")
    return True


from deq2ff.data_utils import reorder_dataset


def test_consecutive_order(args):
    print("\n", "-" * 80, "\n", inspect.currentframe().f_code.co_name)
    """ Dataset """
    train_dataset, val_dataset, test_dataset = rmd17_dataset.get_md_datasets(
        root=args.data_path,
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        return_idx=False,
        order="consecutive",
    )

    print(f"Train dataset: {len(train_dataset)}")
    print(f"Val dataset: {len(val_dataset)}")
    print(f"Test dataset: {len(test_dataset)}")

    # Batches consecutive?
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=True,
    #     num_workers=args.workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=True,
    # )
    # val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True)

    test_dataset = reorder_dataset(test_dataset, args.eval_batch_size)
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True
    )

    last_idx = None
    for step, data in enumerate(test_loader):

        # print(data)
        # print(f'idx: {data.idx}')

        if last_idx is not None:
            assert torch.allclose(data.idx, last_idx + 1), f"{step}"
        last_idx = data.idx

        if step >= 10:
            break

    print(inspect.currentframe().f_code.co_name, "passed!")
    return True


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


@hydra.main(
    config_name="md17",
    config_path="../equiformer/config/equiformer",
    version_base="1.3",
)
def hydra_wrapper(args: DictConfig) -> None:
    # print(train_idx)torch.manual_seed(args.seed)

    set_seed(args.seed)

    # run tests

    test_revisedold_equal_original(args)

    test_revised_dataset_creation(args)

    test_consecutive_order(args)

    test_load_revised_split(args)

    print("\n")
    print("All tests passed!")


if __name__ == "__main__":
    hydra_wrapper()
