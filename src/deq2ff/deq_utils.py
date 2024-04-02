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
        if (step % log_every_step_minor == 0) or (datasplit in ["test", "val"]):
            # log the final fixed point error
            wandb.log({f"abs_fixed_point_error{n}": f_abs_trace[-1].item()}, step=step)
            wandb.log({f"rel_fixed_point_error{n}": f_rel_trace[-1].item()}, step=step)
            # log how many steps it took to reach the fixed point
            wandb.log({f"f_steps_to_fixed_point{n}": len(f_abs_trace)}, step=step)

        if (step % log_every_step_major == 0) or (datasplit in ["test", "val"]):
            # log the fixed point error along the solver trajectory
            # https://github.com/wandb/wandb/issues/3966
            # https://github.com/wandb/wandb/issues/2981#issuecomment-1686868189

            # try to load the table
            table_key = f"fixed_point_error_traj{n}"
            try:
                api = wandb.Api()
                run_id = "EquilibriumEquiFormer" + "/" + wandb.run.id
                run = api.run(run_id)
                artifact = run.logged_artifacts()[0]
                table_old = artifact.get(table_key)
                create_new_table = False
            except Exception as e:
                print(f'Error loading table: {e}')
                print(f'Creating new table for {table_key} (split: {datasplit}).')
                create_new_table = True
            
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

            if create_new_table:
                table_new = wandb.Table(dataframe=data_df)
                wandb.log({table_key: table_new}, step=step)
            
            else:
                # table_old.add_data(...)
                # loop over rows and add them to the table
                for i in range(len(data_df)):
                    table_old.add_data(data_df.iloc[i].values)
                wandb.log({table_key: table_old})
    return

def log_fixed_point_norm(z, step, datasplit=None):
    """Log the norm of the fixed point."""
    if datasplit is None:
        n = ""
    else:
        n = f"_{datasplit}"
    if (step % log_every_step_major == 0) or (datasplit in ["test", "val"]):
        wandb.log({f"fixed_point_norm{n}": z[-1].norm().item()}, step=step)
    return