import torch
import wandb

def log_fixed_point_error(info, step, dataset=None):
    """Log fixed point error to wandb."""
    # absolute fixed point errors along the solver trajectory
    f_abs_trace = info['abs_trace'] 
    f_abs_trace = f_abs_trace.mean(dim=0)[1:]
    # relative fixed point errors along the solver trajectory
    f_rel_trace = info['rel_trace']
    f_rel_trace = f_rel_trace.mean(dim=0)[1:]
    # just log the last value
    if dataset is None:
        n = ''
    else:
        n = f'_{dataset}'
    if len(f_abs_trace) > 0:
        wandb.log({f"abs_fixed_point_error{n}": f_abs_trace[-1].item()}, step=step)
        wandb.log({f"rel_fixed_point_error{n}": f_rel_trace[-1].item()}, step=step)
    return