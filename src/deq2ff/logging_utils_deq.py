import torch
import wandb
import pandas as pd
import os
from typing import Optional, Dict, List

log_every_step_major = 1000
log_every_step_minor = 100


def log_fixed_point_error_trace_table(
    info,
    step,
    datasplit=None,
    data_dicts: List[Dict[str, pd.Series]] = None,
    log_fp_error_traj=True,
):
    """Log fixed point error to wandb using a table."""
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
        #### log error trajectory
        if (step % log_every_step_major == 0) or datasplit in ["test", "val"]:
            # log the fixed point error along the solver trajectory
            # https://github.com/wandb/wandb/issues/3966
            # https://github.com/wandb/wandb/issues/2981#issuecomment-1686868189

            # data
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
            data_df = pd.DataFrame(data_dict)

            table_key = f"fixed_point_error_traj{n}"
            if data_dicts is None:
                # try to load the table from the API
                # NOT RECOMMENDED! This is slow and inefficient.
                try:
                    api = wandb.Api()
                    project = "EquilibriumEquiFormer"
                    run_id = wandb.run.id
                    # V1
                    # run = api.run(project + "/" + run_id)
                    # artifact = run.logged_artifacts()[0]
                    # table = artifact.get(table_key)
                    # V2
                    a = api.artifact(f"{project}/run-{run_id}-{table_key}:latest")
                    table = a.get(table_key)
                    # df_old = pd.DataFrame(data=table.data, columns=table.columns)
                    create_new_table = False
                except Exception as e:
                    print(f"Error loading table: {e}")
                    print(f"Creating new table for {table_key} (split: {datasplit}).")
                    create_new_table = True

                if create_new_table:
                    table = wandb.Table(dataframe=data_df)

                else:
                    # table.add_data(...)
                    # loop over rows and add them to the table
                    for i in range(len(data_df)):
                        try:
                            table.add_data(*data_df.iloc[i].values)
                        except Exception as e:
                            print(f"Error adding data to table: {e}")
                            print(f"data_df.iloc[i].values: {data_df.iloc[i].values}")

                wandb.log({table_key: table}, step=step)
                print(f"Logged table {table_key} (split: {datasplit}) at step {step}.")
                return [data_dict]

            else:
                # data_dicts was passed and is a List[Dict[pd.Series]]
                data_dicts.append(data_dict)
                # merge the data_dicts
                series_concat = {
                    k: pd.concat([d[k] for d in data_dicts], axis=0)
                    for k in data_dicts[0].keys()
                }
                data_df = pd.DataFrame(series_concat)
                table = wandb.Table(dataframe=data_df)
                wandb.log({table_key: table}, step=step)
                print(f"Logged table {table_key} (split: {datasplit}) at step {step}.")
                return data_dicts
        else:
            # do not log anything
            return None


def log_fixed_point_error(
    info, step, datasplit=None, log_trace_freq=None, save_to_file=False
):
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
        # during test and val the step is not updated, i.e. we will never log
        # test and val unless step happens to be a multiple of log_every_step
        # if (step % log_every_step_minor == 0) or (datasplit in ["test", "val"]):
        #     # log the final fixed point error
        #     wandb.log({f"abs_fixed_point_error{n}": f_abs_trace[-1].item()}, step=step)
        #     wandb.log({f"rel_fixed_point_error{n}": f_rel_trace[-1].item()}, step=step)
        #     # log how many steps it took to reach the fixed point
        #     wandb.log(
        #         {f"f_steps_to_fixed_point{n}": info["nstep"][0].mean().item()},
        #         step=step,
        #     )

        # log, but not as table
        if log_trace_freq is None:
            log_trace_freq = log_every_step_major
        if (step % log_trace_freq == 0) or datasplit in ["test", "val"]:
            # print('Logging fixed point error', f"abs_fixed_point_error_traj{n}")
            # log the fixed point error along the solver trajectory
            # https://github.com/wandb/wandb/issues/3966
            _abs = f_abs_trace.detach().cpu().numpy().tolist()
            _rel = f_rel_trace.detach().cpu().numpy().tolist()
            print(f'step={step}, abs trace', datasplit, _abs)
            # print('   _abs', _abs)
            # only log if the list does not contain NaNs or Nones
            if all([x is not None for x in _abs]) and all(
                [x is not None for x in _rel]
            ):
                # print('   _abs', 'not None')
                print(f"Logging fixed point error trajectory. (split: {datasplit} at step {step}).")
                wandb.log(
                    {
                        f"abs_fixed_point_error_traj{n}": _abs,
                        f"rel_fixed_point_error_traj{n}": _rel,
                    },
                    step=step,
                )
                # print(
                #     f"Logged fixed point error trajectory. (split: {datasplit} at step {step}). "
                #     f'Dimension: {len(_abs)}, {info["abs_trace"].mean(dim=0)[1:].shape}'
                # )
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
                            f"abs64_fixed_point_error_traj{n}": _abs64,
                            f"rel64_fixed_point_error_traj{n}": _rel64,
                        },
                        step=step,
                    )
                    # print(
                    #     f"Logged fixed point error trajectory in float64. (split: {datasplit} at step {step}). "
                    #     f'Dimension: {len(_abs64)}, {info["abs_trace64"].mean(dim=0)[1:].shape}'
                    # )

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
