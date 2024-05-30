import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib

project = "EquilibriumEquiFormer"

from src.deq2ff.plotting.style import set_seaborn_style, set_style_after

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

def main(run_id, datasplit="test_fpreuse"):
    # idx_table
    table_key = f"idx_table{datasplit}"
    # table_key = "_idx_table_{_datasplit}"

    # a = artifacts[-1]
    # table = a.get(table_key)
    # df = pd.DataFrame(table)

    api = wandb.Api()
    a = api.artifact(f"{project}/run-{run_id}-{table_key}:latest")
    # apath = a.download()
    table = a.get(table_key)
    df = pd.DataFrame(data=table.data, columns=table.columns)

    # columns: "idx", "e_mae", "f_mae", "nstep"
    # plot the loss per idx
    set_seaborn_style()

    fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="idx", y="e_mae", ax=ax, label="Energy MAE")
    sns.lineplot(data=df, x="idx", y="f_mae", ax=ax, label="Force MAE")
    # ax.set_yscale("log")
    ax.set_xlabel("Index")
    ax.set_ylabel("MAE")
    ax.legend()
    set_style_after(ax)

    # save the plot
    plotname = f"loss_per_idx_{datasplit}.png"
    plotpath = os.path.join(plotfolder, plotname)
    plt.savefig(plotpath)
    print(f"Saved plot to\n {plotpath}")




if __name__ == "__main__":

    # DEQE2FPC fpcof evalbatchsize-1 lr-5e-3 target-ethanol
    # https://wandb.ai/andreas-burger/EquilibriumEquiFormer/runs/m9lxtjgn
    run_id = "m9lxtjgn"
    main(run_id)