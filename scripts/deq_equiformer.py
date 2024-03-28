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

# from logger import FileLogger
from pathlib import Path
from typing import Iterable, Optional

import equiformer.datasets.pyg.md17 as md17_dataset

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
uses equiformer/main_md17.py
"""

import deq2ff
from deq2ff.deq_dp_attention_transformer_md17 import *
from deq2ff.deq_graph_attention_transformer_md17 import *

# silence:
# UserWarning: The TorchScript type system doesn't support instance-level annotations on empty non-base types in `__init__`. 
# Instead, either 1) use a type annotation in the class body, or 2) wrap the type in `torch.jit.Attribute`.
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(config_name="deq", config_path="../equiformer/config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop.
    
    Usage:
        python deq_equiformer.py
        python deq_equiformer.py batch_size=8
        python deq_equiformer.py +machine=vector

    Usage with slurm:
        sbatch scripts/slurm_launcher.slrm deq_equiformer.py +machine=vector
    """

    # args.model_name = "deq_dot_product_attention_transformer_exp_l2_md17"
    # args.deq_mode = True
    # args.num_layers = 2 # 6 -> 1
    # args.meas_force = True 

    args.output_dir = "models/md17/deq-equiformer/test1"

    from equiformer.main_md17 import main

    from deq2ff.logging_utils import init_wandb
    init_wandb(args)

    main(args)


if __name__ == "__main__":


    # TODO try to overfit on tiny subset of data
    # args.train_size = 100

    hydra_wrapper()