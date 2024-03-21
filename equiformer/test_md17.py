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
from logger import FileLogger
from pathlib import Path
from typing import Iterable, Optional

import datasets.pyg.md17 as md17_dataset

import nets
from nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from optim_factory import create_optimizer

from engine import AverageMeter, compute_stats


if __name__ == "__main__":
    from main_md17 import get_args_parser

    parser = argparse.ArgumentParser(
        "Training equivariant networks on MD17", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    ##############################################################################

    """ Dataset """
    train_dataset, val_dataset, test_dataset = md17_dataset.get_md17_datasets(
        root=os.path.join(args.data_path, args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
    )

    print("")
    print("Training set size:   {}".format(len(train_dataset)))
    print("Validation set size: {}".format(len(val_dataset)))
    print("Testing set size:    {}\n".format(len(test_dataset)))

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())
    print("Training set mean: {}, std: {}\n".format(mean, std))

    # since dataset needs random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Data Loader """
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    """ Train """
    for step, data in enumerate(train_loader):
        data = data.to(device)

        print("data", type(data), data.keys)

        # pred_y, pred_dy = model(node_atom=data.z, pos=data.pos, batch=data.batch)

        break
