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

project = "EquilibriumEquiFormer"

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

def main(run_id: str, datasplit: str = "train"):
    api = wandb.Api()
    run = api.run(f'{project}/{run_id}')
    artifacts = run.logged_artifacts()

    print(f'len(artifact): {len(artifacts)}')
    # if len(artifacts) > 1:
    #     main_old(artifacts, datasplit, run_id)
    
    # else:
    #     main_new(artifacts, datasplit, run_id)

    main_new(artifacts, datasplit, run_id)


def main_new(artifacts, datasplit, run_id):

    table_key = f"fixed_point_error_traj_{datasplit}"

    # a = artifacts[-1]
    # table = a.get(table_key)
    # df = pd.DataFrame(table)

    api = wandb.Api()
    a = api.artifact(f'{project}/run-{run_id}-{table_key}:latest')
    # apath = a.download()
    table = a.get(table_key)
    df = pd.DataFrame(data=table.data, columns=table.columns)

    print(f'df: \n{df} ')

    # print(f'df: \n{df}')

    # plot the fixed point error trajectory
    # rel_fixed_point_error_traj on the y-axis
    # solver_step on the x-axis
    # train_step as the hue
    df = df.melt(
        id_vars=["solver_step", "train_step"], value_vars=[f"rel_fixed_point_error_traj_{datasplit}"]
    )
    # print(df)

    sns.lineplot(data=df, x="solver_step", y="value", hue="train_step")
    fname = f"{plotfolder}/fixed_point_error_traj_{datasplit}_{run_id.split('/')[-1]}.png"
    plt.savefig(fname)
    print(f'Saved plot to {fname}')


def main_old(artifacts, datasplit, run_id):
    # indiviual steps are their own tables
    # we need to merge them

    maxtables = 10

    # merge all artifacts
    df = None
    for i, a in enumerate(artifacts):
        table = a.get("fixed_point_error_traj")

        # check if right datasplit
        if table.columns[0].split("_")[-1] != datasplit:
            continue
        
        dict_table = {column: table.get_column(column) for column in table.columns}
        if df is None:
            df = [copy.deepcopy(dict_table)]
        else:
            # append
            df.append(copy.deepcopy(dict_table))

        # if i >= maxtables:
        #     break

        if len(df) >= maxtables:
            break

    # combine the pd.Series in the dict
    df = {k: pd.concat([pd.Series(d[k]) for d in df], axis=0) for k in df[0].keys()}

    df = pd.DataFrame(df)
    # print(f'df: \n{df}')

    # plot the fixed point error trajectory
    # rel_fixed_point_error_traj on the y-axis
    # solver_step on the x-axis
    # train_step as the hue
    df = df.melt(
        id_vars=["solver_step", "train_step"], value_vars=[f"rel_fixed_point_error_traj_{datasplit}"]
    )
    # print(df)

    sns.lineplot(data=df, x="solver_step", y="value", hue="train_step")
    plt.show()


if __name__ == "__main__":

    # DEQ newlogging xpp1auzp
    # FCTPProjectionNorm tlii4hro
    # DEQ noeval inputinjection 6jsxx1x1
    # DEQ ijhtf460
    run_id = "xpp1auzp"
    
    main(run_id, datasplit="train")
