import torch
import wandb
import pandas as pd

from typing import Optional, Dict, List

log_every_step_major = 1000
log_every_step_minor = 100


def log_fixed_point_error(
    info,
    step,
    datasplit=None,
    data_dicts: List[Dict[str, pd.Series]] = None,
    log_fp_error_traj=True,
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
        if (step % log_every_step_minor == 0) or (datasplit in ["test", "val"]):
            # log the final fixed point error
            wandb.log({f"abs_fixed_point_error{n}": f_abs_trace[-1].item()}, step=step)
            wandb.log({f"rel_fixed_point_error{n}": f_rel_trace[-1].item()}, step=step)
            # log how many steps it took to reach the fixed point
            wandb.log({f"f_steps_to_fixed_point{n}": len(f_abs_trace)}, step=step)

        #### log error trajectory
        if log_fp_error_traj:
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
                        print(
                            f"Creating new table for {table_key} (split: {datasplit})."
                        )
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
                                print(
                                    f"data_df.iloc[i].values: {data_df.iloc[i].values}"
                                )

                    wandb.log({table_key: table}, step=step)
                    print(
                        f"Logged table {table_key} (split: {datasplit}) at step {step}."
                    )
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
                    print(
                        f"Logged table {table_key} (split: {datasplit}) at step {step}."
                    )
                    return data_dicts
            else:
                # do not log anything
                return None

        else:
            # log, but not as table
            if (step % log_every_step_major == 0) or datasplit in ["test", "val"]:
                # log the fixed point error along the solver trajectory
                # https://github.com/wandb/wandb/issues/3966
                wandb.log(
                    {
                        f"abs_fixed_point_error_traj{n}": f_abs_trace.detach()
                        .cpu()
                        .numpy()
                        .tolist(),
                        f"rel_fixed_point_error_traj{n}": f_rel_trace.detach()
                        .cpu()
                        .numpy()
                        .tolist(),
                    },
                    step=step,
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
