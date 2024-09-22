import hydra
import wandb
import omegaconf
from omegaconf import OmegaConf
import numpy as np

import os
import sys

import torch

from equiformer.config.paths import ROOT_DIR
from equiformer.optim_factory import scale_batchsize_lr


def fix_args(args: OmegaConf):
    """Fix invalid arg combinations and add runtime information."""
    slurm_job_id = os.environ.get("SLURM_JOB_ID", None)
    args.slurm_job_id = int(slurm_job_id) if slurm_job_id is not None else None
    args = set_gpu_name(args)

    if "test_solver" in args and args.test_solver is not None:
        print("Setting deq f_solver to", args.test_solver)
        args.deq_kwargs.f_solver = args.test_solver

    args = scale_batchsize_lr(args, k=args.get("bsscale", None))

    # allows to load checkpoint with the same name
    if ("evaluate" in args) and args.evaluate:
        if args.wandb_tags is None:
            args.wandb_tags = []
        args.wandb_tags.append("eval")
    if ("eval_speed" in args) and args.eval_speed:
        if args.wandb_tags is None:
            args.wandb_tags = []
        args.wandb_tags.append("speed")
    
    if ("mode" in args) and args.mode != "train":
        if args.wandb_tags is None:
            args.wandb_tags = []
        args.wandb_tags.append(args.mode)

    # rename model to include deq
    if "model_is_deq" in args and args.model_is_deq is True:
        if args.model.name[:3] != "deq":
            args.model.name = f"deq_{args.model.name}"

    # regular Equiformer cannot use fpreuse_test
    else:
        if "equiform_allow_fpreuse" not in args or args.equiform_allow_fpreuse is False:
            if "fpreuse_test" in args:
                args.fpreuse_test = False

    # if we use fpreuse, we need to make sure that the test set is consecutive across batches
    if "fpreuse_test" in args and args.fpreuse_test:
        if args.datasplit not in ["fpreuse_overlapping", "fpreuse_ordered"]:
            print(
                'Warning: fpreuse_test is set, but datasplit is not "fpreuse_overlapping" or "fpreuse_ordered". Setting datasplit to "fpreuse_overlapping"'
            )
            args.datasplit = "fpreuse_overlapping"
    # if we use contrastive loss, the train set needs to be consecutive within a batch
    # deprecated?
    if isinstance(args.contrastive_loss, str) and args.contrastive_loss.endswith(
        "ordered"
    ):
        if args.datasplit not in ["fpreuse_ordered"]:
            print(
                'Warning: contrastive_loss is set, but datasplit is not "fpreuse_ordered". Setting datasplit to "fpreuse_ordered"'
            )
            args.datasplit = "fpreuse_ordered"

    # model with just energy prediction
    if "noforcemodel" in args and args.noforcemodel is True:
        if args.model.name[-7:] != "noforce":
            args.model.name = f"{args.model.name}_noforce"
        args.meas_force = False

    # set names
    if args.wandb_run_name is None:
        args.wandb_run_name = args.model.name
    # for human readable names
    wandb_run_name = name_from_config(args)
    # for checkpoint names
    args.checkpoint_wandb_name = name_from_config(args, is_checkpoint_name=True)
    # we can only set the name after the checkpoint name is set
    args.wandb_run_name = wandb_run_name

    return args


def init_wandb(args: OmegaConf, project="Equi2"):
    """init shared across all methods"""

    args = fix_args(args)

    if args.wandb == False:
        # wandb.init(mode="disabled")
        os.environ["WANDB_DISABLED"] = "true"

    # to dict
    # omegaconf.errors.InterpolationResolutionError: Recursive interpolation detected
    args = OmegaConf.structured(OmegaConf.to_yaml(args))

    wandb.require("core")
    # wandb.run.name = name_from_config(args)
    wandb.init(
        group=args.wandb_group,
        project=project,
        # entity="andreas-burger",
        name=args.wandb_run_name,
        config=OmegaConf.to_container(args, resolve=True),
        tags=args.wandb_tags,
        # reinit=True,
        # settings=wandb.Settings(start_method="fork")
    )

    # add runid to config
    wandb.config.update({"run_id": wandb.run.id})

    print("wandb group:", args.wandb_group, "==", wandb.run.group)
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


# allows to load checkpoint with the same name
IGNORE_OVERRIDES = [
    "resume_from_checkpoint",
    "save_checkpoint_after_test",
    "max_checkpoints",
    "save_final_checkpoint",
    "save_best_test_checkpoint",
    "save_best_val_checkpoint",
    "output_dir",
    "logging_dir",
    "checkpointing_steps",
    "override_dirname",
    "wandb_project",
    "wandb_entity",
    "wandb_run_name",
    "wandb_tags",
    "watch_model",
    "assert_checkpoint",
    "evaluate",
    "num_train_epochs",
    "machine",
    "basemodel",
    "wandb_group",
    "wandb",
]

# some stuff is not relevant for the checkpoint
# e.g. inference kwargs
IGNORE_OVERRIDES_CHECKPOINT = [
    "deq_kwargs_test",
    "deq_kwargs_eval",
    "deq_kwargs_fpr",
    "deq_kwargs_eval_fpr",
    "fpreuse_f_tol",
    "fpreuse_test",
    "eval_batch_size",
    "wandb_tags",
    "evaluate",
    "test_w_eval_mode",
    "testwevalmode",
    "test_w_grad",
    "testwgrad",
    "test_max_iter",
    "test_patch_size",
    "datasplit",
    "test_patches",
    # "trial",
    # "+trial",
    # target is defined in the path
    # "target",
    "inf",
    "fulleval",
    "test_solver",
]

REPLACE = {
    "deq_dot_product_attention_transformer_exp_l2_md17": "DEQE1",
    "dot_product_attention_transformer_exp_l2_md17": "E1",
    "deq_equiformer_v2_oc20": "pDEQs",  # ChangeS DEQc
    "equiformer_v2_oc20": "pEs",  # ChangeS E2 OC20
    "deq_equiformer_v2_md17": "pDEQs",  # ChangeS DEQc
    "equiformer_v2_md17": "pEs",  # ChangeS E2 MD17
    # other stuff
    "dot_product": " dp",
    "use-deq": "",
    "deq": "DEQ",
    "initzfromenc-True": "V1",
    "initzfromenc-False": "V2",
    "deqkwargs": "",
    "modelkwargs": "",
    "model": "",
    "_attention": "",
    "_transformer": "",
    "_md17": "",
    "_dp": "",
    # "dpa": "Equiformer",
    # "dp": "Equiformer",
    "_l2": "",
    "_graph_nonlinear": " GraphNonLinear",
    "_decprojhead": "",
    "_decproj-": " dec-",
    "_evalmode-False": " noeval",
    "_revised": " revised",
    "userevisedsplits-True": " revisedsplit",
    "_fpreusetest-True": " FPreuse",
    "_torchDEQ": " ",
    "-True": " ",
    "normtype-": "",
    "cfg-": "",
    "fsolver-": "",
    "normln": "ln",
    "_": " ",
    "preset-": "",
    "catinjection-False": "add-inj",
    " minimalaDEQ minimala": "min",
    "DEQblock-": "block-",
    "name-minimalEquiformer": "",
    "pathnorm-": "-",
    "EquivariantLayerNormV2": "ELN",
    "ParameterList": "PL",
    "DEQ dp use-DEQ": "DEQ E1",
    # Equiformer v2
    "equiformer v2equiformer v2 use-DEQ": "DEQE2",
    "equiformer v2equiformer v2": "E2",
    "equiformer v2 use-DEQ": "DEQE2",
    "DEQ DEQ": "DEQ",
    "DEQ equiformer v2DEQE2": "DEQE2",
    "DEQ  dp use-DEQ": "DEQ E1",
    "torchDEQnorm": "",
    "  ": " ",
    # "alphadrop": "ad",
    # "pathdrop": "pd",
    # ignore defaults
    "targetaspirin": "",
    "seed1": "",
    "fstopmode-": "",
    "target-": "",
    "test-": "t-",
    "layernorm": "ln",
    "dname-": "",
    # "lntype-layernorm": "type-ln",
}


def name_from_config(args: omegaconf.DictConfig, is_checkpoint_name=False) -> str:
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
        # print(f'Overrides: {args.override_dirname}')
        if args.override_dirname:
            for arg in args.override_dirname.split(","):
                # make sure we ignore some overrides
                if np.any([ignore in arg for ignore in IGNORE_OVERRIDES]):
                    continue
                if is_checkpoint_name:
                    if np.any(
                        [ignore in arg for ignore in IGNORE_OVERRIDES_CHECKPOINT]
                    ):
                        continue
                override = arg.replace("+", "").replace("_", "")
                override = override.replace("=", "-").replace(".", "")
                # override = override.replace("deqkwargstest", "")
                override = override.replace("deqkwargs", "").replace("model", "")
                override_names += " " + override
    except Exception as error:
        print("\nname_from_config() failed:", error)
        print("args:", args)
        raise error
    # logger.info("name_from_config() mname: %s, override_names: %s", mname, override_names)
    _name = mname + override_names
    for key, value in REPLACE.items():
        _name = _name.replace(key, value)
    # more complex replacements
    # alphadrop-01 pathdrop-005 usevariationalalphadrop usevariationalpathdrop
    if "alphadrop" in _name and "usevariationalalphadrop" in _name:
        _name = _name.replace(" usevariationalalphadrop", f"")
        _name = _name.replace("alphadrop", f"varalphadrop")
    if "pathdrop" in _name and "usevariationalpathdrop" in _name:
        _name = _name.replace(" usevariationalpathdrop", f"")
        _name = _name.replace("pathdrop", f"varpathdrop")
    if "model_is_deq" in args and args.model_is_deq is False:
        _name = _name.replace("numlayers-4", f"")
        if "pathdrop-005" in _name and "alphadrop-01" in _name:
            _name = _name.replace("pathdrop-005", f"")
            _name = _name.replace("alphadrop-01", f"dd")
    if "aspirin" in _name:
        _name = _name.replace("aspirin", f"")
    # done
    print(f"Name{' checkpoint' if is_checkpoint_name else ''}: {_name}")
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


# def set_wandb_name(args, initial_global_step, global_step=None):
#     """Set wandb.run.name."""
#     try:
#         gpu_name = torch.cuda.get_device_name(0)
#         gpu_name = (
#             gpu_name.replace("NVIDIA", "").replace("GeForce", "").replace(" ", "")
#         )
#         run_name = name_from_config(args)
#         run_name += f"-{gpu_name}-{initial_global_step}"
#         if (global_step is not None) and (global_step != initial_global_step):
#             run_name += f"-{global_step}"
#         wandb.run.name = run_name
#     except:
#         pass

# replace old keys with new keys (due to renaming parts of the model)
old_to_new_keys = {
    "blocks.0.ga": "blocks.0.graph_attention",
    "blocks.1.ga": "blocks.1.graph_attention",
    "blocks.2.ga": "blocks.2.graph_attention",
    "blocks.3.ga": "blocks.3.graph_attention",
    "blocks.4.ga": "blocks.4.graph_attention",
    "blocks.5.ga": "blocks.5.graph_attention",
    "blocks.6.ga": "blocks.6.graph_attention",
    "blocks.7.ga": "blocks.7.graph_attention",
    "blocks.8.ga": "blocks.8.graph_attention",
}
