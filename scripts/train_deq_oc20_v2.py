import copy
import logging
import os
import sys
import time
from pathlib import Path

import submitit

from ocpmodels.common import distutils
from ocpmodels.common.flags import flags
from ocpmodels.common.registry import registry
from ocpmodels.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_imports,
    setup_logging,
)

# equiformer_v2
import equiformer_v2.nets
import equiformer_v2.oc20.trainer
import equiformer_v2.oc20.trainer.dist_setup

# deq_equiformer_v2
from deq2ff.deq_equiformer_v2.deq_equiformer_v2_oc20 import DEQ_EquiformerV2_OC20
from deq2ff.logging_utils import init_wandb
from deq2ff.oc20runner import Runner

# in PyG version <= 2.0.4
# UserWarning: An output with one or more elements was resized since it had shape [528],
# which does not match the required output shape [176, 3].
# This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements.
# You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0)
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Original config:
# https://github.com/FAIR-Chem/fairchem/blob/v0.1.0/ocpmodels/common/flags.py
# https://github.com/atomicarchitects/equiformer_v2/blob/main/oc20/configs/s2ef/2M/equiformer_v2/equiformer_v2_N%4012_L%406_M%402.yml


import hydra
import wandb
import omegaconf
from omegaconf import DictConfig, OmegaConf


@hydra.main(
    config_name="oc20", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper_oc20(args: DictConfig) -> None:
    """Run training loop.

    Usage:
        python deq_equiformer.py
        python deq_equiformer.py batch_size=8
        python deq_equiformer.py +machine=vector

    Usage with slurm:
        sbatch scripts/slurm_launcher.slrm deq_equiformer.py +machine=vector
    """

    init_wandb(args, project=args.logger.project)
    print("checkpoint_name:", args.checkpoint_wandb_name)

    # turn args into dictionary for compatibility with ocpmodels
    # args: omegaconf.dictconfig.DictConfig -> dict
    args = OmegaConf.to_container(args, resolve=True)
    runner = Runner()
    runner(args)
    runner.run_task()


if __name__ == "__main__":
    setup_logging()

    # ocpmodels
    # parser = flags.get_parser()
    # args, override_args = parser.parse_known_args()
    # config = build_config(args, override_args)

    # use hydra instead of ocpmodels
    hydra_wrapper_oc20()
