import hydra
import wandb
import omegaconf
from omegaconf import OmegaConf
import numpy as np

import os
import sys

import torch

from equiformer.config.paths import ROOT_DIR


def fix_args(args: OmegaConf):
    args.slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    args = set_gpu_name(args)

    if 'dec_proj' in args.model_kwargs:
        if args.model_kwargs.dec_proj is not None:
            if  "decprojhead" not in args.model_name:
                args.model_name = f"decprojhead_{args.model_name}"

    if 'model_is_deq' in args:
        if args.model_is_deq is True:
            if args.model_name[:3] != "deq":
                args.model_name = f"deq_{args.model_name}"

    if 'noforcemodel' in args:
        if args.noforcemodel is True:
            if args.model_name[-7:] != "noforce":
                args.model_name = f"{args.model_name}_noforce"
            args.meas_force = False

    if args.wandb_run_name is None:
        # args.wandb_run_name = args.data_path.split("/")[-1]
        model_name = args.model_name.split("_attention_transformer")
        model_name = model_name[0] + model_name[-1]
        model_name = model_name.replace("_exp_l2", "")
        args.wandb_run_name = model_name
    args.wandb_run_name = name_from_config(args)
    
    return args

def init_wandb(args: OmegaConf):
    """init shared across all methods"""

    args = fix_args(args)

    if args.wandb == False:
        # wandb.init(mode="disabled")
        os.environ["WANDB_DISABLED"] = "true"

    # to dict
    # omegaconf.errors.InterpolationResolutionError: Recursive interpolation detected
    args = OmegaConf.structured(OmegaConf.to_yaml(args))
    # args = OmegaConf.create(args)
    args_wandb = OmegaConf.to_container(args, resolve=True)
    # print("args passed to wandb:\n", args_wandb)

    # wandb.run.name = name_from_config(args)
    wandb.init(
        group=args.wandb_group,
        project="EquilibriumEquiFormer",
        # entity="andreas-burger",
        name=args.wandb_run_name,
        config=args_wandb,
        # reinit=True,
    )

    print("wandb run name:", args.wandb_run_name)
    print("wandb run id:", wandb.run.id)
    return wandb.run.id


def final_logging(args):
    """Log final information."""
    args.slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    # for name, value in os.environ.items():

    if args.slurm_job_id is not None:
        # find file in slurm-<job_id>.out in ROOT_DIR
        slurm_file = f"slurm-{args.slurm_job_id}.out"
        slurm_file = os.path.join(ROOT_DIR, slurm_file)
        if os.path.exists(slurm_file):
            wandb.save(slurm_file, base_path=ROOT_DIR)
            print(f"Saved {slurm_file} to wandb.")
        else:
            print(f"Could not find {slurm_file}.")


IGNORE_OVERRIDES = [
    "resume_from_checkpoint",
    "output_dir",
    "logging_dir",
    "checkpointing_steps",
    "override_dirname",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "num_train_epochs",
    "machine",
    "basemodel",
    "wandb_group",
]

REPLACE = {
    "dot_product": "dp",
    "deq": "DEQ",
    "initzfromenc-True": "V1",
    "initzfromenc-False": "V2",
    "deqkwargs": "",
    "modelkwargs": "",
    "_attention_transformer": "",
    "_md17": "",
    "_dp": "",
    "_l2": "",
    "_graph_nonlinear": " GraphNonLinear",
    "_decprojhead": "",
    "_decproj-": " ",
    "_evalmode-False": " noeval",
}


def name_from_config(args: omegaconf.DictConfig) -> str:
    """Generate a name for the model based on the config.
    Name is intended to be used as a file name for saving checkpoints and outputs.
    """
    try:
        # model name format:
        # deq_dot_product_attention_transformer_exp_l2_md17
        # deq_graph_attention_transformer_nonlinear_l2_md17
        mname = args.wandb_run_name
        # override format: 'pretrain_dataset=bridge,steps=10,use_wandb=False'
        override_names = ""
        if args.override_dirname:
            for arg in args.override_dirname.split(","):
                # make sure we ignore some overrides
                if np.any([ignore in arg for ignore in IGNORE_OVERRIDES]):
                    continue
                else:
                    override = arg.replace("+", "").replace("_", "")
                    override = override.replace("=", "-").replace(".", "")
                    override = override.replace("deqkwargs", "")
                    override_names += "_" + override
    except Exception as error:
        print("\nname_from_config() failed:", error)
        print("args:", args)
        raise error
    # logger.info("name_from_config() mname: %s, override_names: %s", mname, override_names)
    _name = mname + override_names
    for key, value in REPLACE.items():
        _name = _name.replace(key, value)
    return _name


def set_gpu_name(args):
    """Set wandb.run.name."""
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_name = (
            gpu_name.replace("NVIDIA", "").replace("GeForce", "").replace(" ", "")
        )
        args.gpu_name = gpu_name
    except:
        pass
    return args


def set_wandb_name(args, initial_global_step, global_step=None):
    """Set wandb.run.name."""
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_name = (
            gpu_name.replace("NVIDIA", "").replace("GeForce", "").replace(" ", "")
        )
        run_name = name_from_config(args)
        run_name += f"-{gpu_name}-{initial_global_step}"
        if (global_step is not None) and (global_step != initial_global_step):
            run_name += f"-{global_step}"
        wandb.run.name = run_name
    except:
        pass
