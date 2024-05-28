
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
# torch_geometric/data/collate.py:150: UserWarning: An output with one or more elements was resized since it had shape, which does not match the required output shape
import warnings

warnings.filterwarnings("ignore", category=UserWarning)



def main(args):
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

        train_dataset, val_dataset, test_dataset, all_dataset = md_all.get_md_datasets(
            root=args.data_path,
            dataset_arg=args.target,
            dname=args.dname,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=None,
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

    """ Data Loader """
    # We don't need to shuffle because either the indices are already randomized
    # or we want to keep the order
    # we just keep the shuffle option for the sake of consistency with equiformer
    # if args.datasplit in ["fpreuse_ordered"]:
    #     shuffle = False
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
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
        from deq2ff.data_utils import reorder_dataset

        test_dataset = reorder_dataset(test_dataset, args.eval_batch_size)
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True
    )


    """ Dryrun of forward pass for testing """
    first_batch = next(iter(train_loader))

    data = first_batch.to(device)

    print('')
    print("data:", data.idx)

    try:
        print("old_idx:", data.old_idx)
    except AttributeError:
        pass



@hydra.main(config_name="md17", config_path="../equiformer/config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    from deq2ff.logging_utils import init_wandb
    
    args.wandb = False
    args.dname = "md17" # md17, rmd17, rmd17og
    args.datasplit = "consecutive_all"
    args.shuffle = False

    init_wandb(args)

    # args: omegaconf.dictconfig.DictConfig -> dict
    # args = OmegaConf.to_container(args, resolve=True)

    main(args)


if __name__ == "__main__":
    hydra_wrapper()