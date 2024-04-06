import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib

"""
Fixed point convergence
abs_trace over forward-solver-iteration-steps
https://colab.research.google.com/drive/12HiUnde7qLadeZGGtt7FITnSnbUmJr-I?usp=sharing#scrollTo=V5Zff4FHqR5d
"""

entity = "andreas-burger"
project = "EquilibriumEquiFormer"

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

# columns = ['abs', 'rel', 'solver_step', 'train_step']

def main(run_id: str, datasplit: str = "train", error_type="abs"):
    # https://github.com/wandb/wandb/issues/3966

    artifact_name = f"{error_type}_fixed_point_error_traj_{datasplit}"
    alias = "latest"

    api = wandb.Api()
    run = api.run(project + "/" + run_id)

    # artifact = run.logged_artifacts()
    # print(f"len(artifact): {len(artifact)}")
    # artifact = artifact[-1]
    # table = artifact.get(artifact_name)
    # dict_table = {column: table.get_column(column) for column in table.columns}
    # df = pd.DataFrame(dict_table)


    # run = wandb.init()
    # artifact = run.use_artifact(f'{entity}/{project}/{artifact_name}:{alias}')
    # artifact_table = artifact.get(artifact_name)
    # my_df = pd.DataFrame(data=artifact_table.data, columns=artifact_table.columns)
    # wandb.finish()

    # metrics_dataframe = run.history()

    history = run.scan_history()
    losses = [[row[artifact_name], row["_step"]] for row in history if artifact_name in row.keys()]
    # losses = [[r, s, [*range(len(r))]] for r, s in losses]
    losses = [
        {
            error_type: pd.Series(r),
            "train_step": pd.Series([s] * len(r)),
            "solver_step": pd.Series(range(len(r)))
        }
        for r, s in losses
    ]
    losses_concat = {k: pd.concat([d[k] for d in losses], axis=0) for k in losses[0].keys()}
    # print(f"losses_concat: {losses_concat}")
    df = pd.DataFrame(losses_concat)

    # dataframe with three colums: error_type, train_step, solver_step
    print(df)

    # plot: x=solver_step, y=error_type, hue=train_step
    sns.lineplot(data=df, x="solver_step", y=error_type, hue="train_step")
    plt.xlabel("Fixed-point solver step")
    plt.ylabel(f"Fixed-point error ({error_type})")
    fname = (
        f"{plotfolder}/fixed_point_error_traj_{datasplit}_{run_id.split('/')[-1]}.png"
    )
    plt.savefig(fname)
    print(f"Saved plot to {fname}")

if __name__ == "__main__":

    # Tanh: ii3gls8d
    run_id = "ii3gls8d"

    main(run_id, datasplit="train")