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


import hydra
import wandb
import omegaconf
from omegaconf import DictConfig, OmegaConf

from ocpmodels.common.registry import registry



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
                # checkpoint to load
                checkpoint_path=config.get("checkpoint_path", None),
                load_checkpoint=config.get("load_checkpoint", True),
                checkpoint_wandb_name=config.get("checkpoint_wandb_name", None),
                checkpoint_name=config.get("checkpoint.pt", None),  # Provide
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
                deq_kwargs_eval=config.get("deq_kwargs_eval", {}),
                deq_kwargs_fpr=config.get("deq_kwargs_fpr", {}),
                # test_w_eval_mode
            )
            # overwrite mode
            if config.get("compute_stats", False):
                config["mode"] = "compute_stats"
            self.task = registry.get_task_class(config["mode"])(self.config)
            print("Running task: ", self.task.__class__.__name__)
            self.task.setup(self.trainer)
            
        finally:
            pass
            # if args.distributed:
            #     distutils.cleanup()
    
    def run_task(self):
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

    # def checkpoint(self, *args, **kwargs):
    #     new_runner = Runner()
    #     self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
    #     self.config["checkpoint"] = self.task.chkpt_path
    #     self.config["timestamp_id"] = self.trainer.timestamp_id
    #     if self.trainer.logger is not None:
    #         self.trainer.logger.mark_preempting()
    #     return submitit.helpers.DelayedSubmission(new_runner, self.config)

def get_OC20runner(args: DictConfig) -> Runner:
    runner = Runner()
    runner(args)
    return runner

# atomic numbers to chemical symbols
# https://github.com/sugarlabs/periodic-table/blob/master/periodic_elements.py#L24C1-L33C50
elements = [
    " ", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Uut",
    "Fl", "Uup", "Lv", "Uus", "Uuo"
]
# the elements used to build OC20 according to
# https://discuss.opencatalystproject.org/t/which-elements-used-to-build-oc20/290/2
valid_symbols = ['Ag', 'Y', 'N', 'Fe', 'Zr', 'Zn', 'Cu', 'Pb', 'S', 'Pt', 'W', 'P', 'Mo', 'Au', 'Ge', 'Ta']


@hydra.main(
    config_name="oc20", config_path="../../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper_oc20(args: DictConfig) -> None:
    from deq2ff.logging_utils import fix_args_set_name
    import numpy as np
    
    args.wandb = False
    args.optim.batch_size = 1
    args.logger = "dummy"
    
    args = fix_args_set_name(args)

    # turn args into dictionary for compatibility with ocpmodels
    # args: omegaconf.dictconfig.DictConfig -> dict
    args = OmegaConf.to_container(args, resolve=True)
    
    reference = get_OC20runner(args)
    reference = reference.trainer
    
    print(" ")
    print("-"*100)
    
    # print("Trainer config:")
    # print(yaml.dump(reference.config))
    # print([k for k, v in reference.config["cmd"].items() if "checkpoint" in k])
    
    print("checkpoint_dir:", reference.config["cmd"]["checkpoint_dir"])
    print("checkpoint_name:", reference.config["cmd"]["checkpoint_name"])
    print("checkpoint_wandb_name:", reference.config["cmd"]["checkpoint_wandb_name"])
    print("checkpoint_path:", reference.config["cmd"]["checkpoint_path"])
    
    checkpoint = reference.config["cmd"]["checkpoint_dir"]
    checkpoint += "best_checkpoint.pt"
    
    print(" ")
    print("-"*100)
    train_loader = reference.train_loader
    train_loader_iter = iter(train_loader)
    batch = next(train_loader_iter)
    sample = batch[0]
    print("Sample:", sample)
    print("Type:", type(sample))
    print("Pos shape:", sample.pos.shape)
    print("Force shape:", sample.force.shape)
    print("Cell shape:", sample.cell.shape)
    print("Atomic numbers shape:", sample.atomic_numbers.shape)
    
    atomic_numbers = [elements[int(num.item())] for num in sample.atomic_numbers]
    print("Atomic symbols:", atomic_numbers)
    print("Unique symbols:", set(atomic_numbers))
    print("All accounted for:", np.all(np.isin(atomic_numbers, valid_symbols)))


    
    
if __name__ == "__main__":
    hydra_wrapper_oc20()