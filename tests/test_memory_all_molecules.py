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
from omegaconf import DictConfig

ModelEma = ModelEmaV2

# silence:
# UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`.
# Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(config_name="md17", config_path="config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop.

    Usage:
        python deq_equiformer.py
        python deq_equiformer.py batch_size=8
        python deq_equiformer.py +machine=vector

    Usage with slurm:
        sbatch scripts/slurm_launcher.slrm deq_equiformer.py +machine=vector

    To reprocude the paper results:
        python deq_equiformer.py input_irreps='64x0e' weight_decay=1e-6 number_of_basis=32 energy_weight=1 force_weight=80
    """

    from equiformer.datasets.pyg.md_all import MDAll

    all_molecules = [m for dsets, mols in MDAll.molecule_files for m in mols]
    print(all_molecules)

    for dsets, mols in MDAll.molecule_files:
        for mol in mols:
            args.target = mol
            args.dname = dsets

    from deq2ff.logging_utils import init_wandb

    init_wandb(args)

    from equiformer.main_md17 import main

    main(args)
