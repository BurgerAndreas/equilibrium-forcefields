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

class Runner(submitit.helpers.Checkpointable):
    def __init__(self):
        self.config = None

    def __call__(self, config):
        setup_logging()
        self.config = copy.deepcopy(config)

        # if args.distributed:
        #     # distutils.setup(config)
        #     oc20.trainer.dist_setup.setup(config)

        try:
            setup_imports()
            print(f'Initializing trainer: {config["trainer"]}')
            self.trainer = registry.get_trainer_class(config.get("trainer", "energy"))(
                task=config["task"],
                model=config["model"],
                dataset=config["dataset"],
                optimizer=config["optim"],
                identifier=config["identifier"],
                timestamp_id=config.get("timestamp_id", None),
                # checkpointing
                run_dir=config.get("run_dir", "./"),
                checkpoint_path=config.get("checkpoint_path", None), # checkpoint to load
                checkpoint_wandb_name=config.get("checkpoint_wandb_name", None), 
                checkpoint_name=config.get("checkpoint.pt", None), # Provide
                assert_checkpoint=config.get("assert_checkpoint", False),
                is_debug=config.get("is_debug", False),
                print_every=config.get("print_every", 10),
                seed=config.get("seed", 0),
                logger=config.get("logger", "tensorboard"),
                local_rank=config["local_rank"],
                amp=config.get("amp", False),
                cpu=config.get("cpu", False),
                slurm=config.get("slurm", {}),
                noddp=config.get("noddp", False),
                # added
                val_max_iter=config.get("val_max_iter", -1),
                model_is_deq=config.get("model_is_deq", False),
                deq_kwargs=config.get("deq_kwargs", {}),
                # test_w_eval_mode
            )
            # overwrite mode
            if config.get("compute_stats", False):
                config["mode"] = "compute_stats"
            self.task = registry.get_task_class(config["mode"])(self.config)
            print('Running task: ', self.task.__class__.__name__)
            self.task.setup(self.trainer)
            start_time = time.time()
            print(f"Starting task: {self.task.__class__.__name__}")
            self.task.run()
            distutils.synchronize()
            if distutils.is_master():
                time_total = time.time() - start_time
                logging.info(
                    f"Total time taken: {time_total}s ({time_total/3600:.2f}h)"
                )
                # wandb.log({"total_time": time_total})
        finally:
            pass
            # if args.distributed:
            #     distutils.cleanup()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        return submitit.helpers.DelayedSubmission(new_runner, self.config)


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

    init_wandb(args, project="oc20-ev2")
    print("checkpoint_name:", args.checkpoint_wandb_name)

    # turn args into dictionary for compatibility with ocpmodels
    # args: omegaconf.dictconfig.DictConfig -> dict
    args = OmegaConf.to_container(args, resolve=True)
    Runner()(args)


if __name__ == "__main__":
    setup_logging()

    # ocpmodels
    # parser = flags.get_parser()
    # args, override_args = parser.parse_known_args()
    # config = build_config(args, override_args)

    # use hydra instead of ocpmodels
    hydra_wrapper_oc20()
