import torch
import wandb
import pandas as pd

log_every_step_major = 1000
log_every_step_minor = 100


def log_fixed_point_error(info, step, datasplit=None):
    """Log fixed point error to wandb."""
    # absolute fixed point errors along the solver trajectory
    f_abs_trace = info["abs_trace"]
    f_abs_trace = f_abs_trace.mean(dim=0)[1:]
    # relative fixed point errors along the solver trajectory
    f_rel_trace = info["rel_trace"]
    f_rel_trace = f_rel_trace.mean(dim=0)[1:]
    # just log the last value
    if datasplit is None:
        n = ""
    else:
        n = f"_{datasplit}"

    if len(f_abs_trace) > 0:
        if step % log_every_step_minor == 0:
            # log the final fixed point error
            wandb.log({f"abs_fixed_point_error{n}": f_abs_trace[-1].item()}, step=step)
            wandb.log({f"rel_fixed_point_error{n}": f_rel_trace[-1].item()}, step=step)
        if step % log_every_step_major == 0:
            # log the fixed point error along the solver trajectory
            # https://github.com/wandb/wandb/issues/3966
            data_dict = {
                f"abs_fixed_point_error_traj{n}": pd.Series(
                    f_abs_trace.detach().cpu().numpy()
                ),
                f"rel_fixed_point_error_traj{n}": pd.Series(
                    f_rel_trace.detach().cpu().numpy()
                ),
                "solver_step": pd.Series(range(len(f_abs_trace))),
                "train_step": pd.Series([step] * len(f_abs_trace)),
            }
            table = wandb.Table(dataframe=pd.DataFrame(data_dict))
            wandb.log({"fixed_point_error_traj": table}, step=step)
            # get the values later
            # api = wandb.Api()
            # run = api.run("run_id")
            # artifact = run.logged_artifacts()[0]
            # table = artifact.get("fixed_point_error_traj")
            # dict_table = {column: table.get_column(column) for column in table.columns}
            # df = pd.DataFrame(dict_table)
    return

def log_fixed_point_norm(z, step, datasplit=None):
    """Log the norm of the fixed point."""
    if datasplit is None:
        n = ""
    else:
        n = f"_{datasplit}"
    if step % log_every_step_major == 0:
        wandb.log({f"fixed_point_norm{n}": z[-1].norm().item()}, step=step)
    return