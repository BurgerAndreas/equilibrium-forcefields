import torch
import wandb
import pandas as pd
import os
from typing import Optional, Dict, List

log_every_step_major = 1000
log_every_step_minor = 100


def log_fixed_point_error(
    info, step, datasplit=None, split=None, log_trace_freq=None, save_to_file=False
):
    """Log fixed point error to wandb.

    datasplit: 'train', 'val', 'test' will be manually appended to the wandb log key "key_datasplit" ("nstep_train)
    split: 'train', 'val', 'test' will be manually appended to the wandb log key "split/key"
    """
    # [B, traj_len] -> [traj_len]
    # TorchDEQ stopping criterion is .max() < tol, not .mean()
    # absolute fixed point errors along the solver trajectory
    f_abs_trace = info["abs_trace"]
    f_abs_trace, maxind = f_abs_trace.max(dim=0)
    # relative fixed point errors along the solver trajectory
    f_rel_trace = info["rel_trace"]
    f_rel_trace, maxind = f_rel_trace.max(dim=0)
    # just log the last value
    if datasplit is None:
        n = ""
    else:
        n = f"_{datasplit}"
    if split is None:
        npre = ""
    else:
        npre = f"{split}/"

    if len(f_abs_trace) > 0:

        # log, but as a list instead of table
        if log_trace_freq is None:
            log_trace_freq = log_every_step_major
        if (
            (step % log_trace_freq == 0)
            or datasplit in ["test", "val"]
            or split in ["test", "val"]
        ):
            # print('Logging fixed point error', f"abs_fixed_point_error_traj{n}")
            # log the fixed point error along the solver trajectory
            # https://github.com/wandb/wandb/issues/3966
            _abs = f_abs_trace.cpu().numpy().tolist()
            _rel = f_rel_trace.cpu().numpy().tolist()

            # only log if the list does not contain NaNs or Nones
            if all([x is not None for x in _abs]) and all(
                [x is not None for x in _rel]
            ):
                # print('   _abs', 'not None')

                wandb.log(
                    {
                        f"{npre}abs_fixed_point_error_traj{n}": _abs,
                        f"{npre}rel_fixed_point_error_traj{n}": _rel,
                    },
                    step=step,
                    # split=split,
                )

            # log again in float64 format
            if "abs_trace64" in info:
                _abs64 = (
                    info["abs_trace64"].mean(dim=0)[1:].detach().cpu().numpy().tolist()
                )
                _rel64 = (
                    info["rel_trace64"].mean(dim=0)[1:].detach().cpu().numpy().tolist()
                )
                # only log if the list does not contain NaNs or Nones
                if all([x is not None for x in _abs]) and all(
                    [x is not None for x in _rel]
                ):
                    wandb.log(
                        {
                            f"{npre}abs64_fixed_point_error_traj{n}": _abs64,
                            f"{npre}rel64_fixed_point_error_traj{n}": _rel64,
                        },
                        step=step,
                        # split=split,
                    )

    return None


def log_fixed_point_norm(z, step, datasplit=None, name="fixed_point"):
    """Log the norm of the fixed point."""
    if datasplit is None:
        n = ""
    else:
        n = f"_{datasplit}"
    if (step % log_every_step_major == 0) or datasplit in ["test", "val"]:
        # TODO replace with https://pytorch.org/docs/stable/generated/torch.linalg.norm.html
        wandb.log({f"{name}_norm{n}": z[-1].norm().item()}, step=step)
        wandb.log({f"{name}_mean{n}": z[-1].mean().item()}, step=step)
        wandb.log({f"{name}_std{n}": z[-1].std().item()}, step=step)
    return


def check_values(a, name, step, datasplit=None):
    # check for infinite values
    if not torch.isfinite(a).all():
        print(f"{name} has infinite values")
        return False
    # check for nan values
    if torch.isnan(a).any():
        print(f"{name} has nan values")
        return False
    return True


def print_values(a, name, step=None, datasplit=None, log=False, before=None):
    if os.environ.get("PRINT_VALUES", 0) == "1":
        if before is not None:
            print(before)
        _s = f"Step {step}: " if step is not None else ""
        _d = f" ({datasplit})" if datasplit is not None else ""
        print(
            f"{_s}{name}{_d}:"
            f" min: {a.min().item()}, max: {a.max().item()}, mean: {a.mean().item()}, norm: {a.norm().item()}"
            f" isinf: {torch.isinf(a).any()}, isnan: {torch.isnan(a).any()}"
        )
    # log to wandb
    if log and step is not None:
        _name = name + (f"_{datasplit}" if datasplit is not None else "")
        wandb.log(
            {
                f"{_name}_min": a.min().item(),
                f"{_name}_max": a.max().item(),
                f"{_name}_mean": a.mean().item(),
                f"{_name}_norm": a.norm().item(),
                f"{_name}_isfinite": torch.isfinite(a).all().item(),
                f"{_name}_isnan": torch.isnan(a).any().item(),
            },
            step=step,
        )
    return
