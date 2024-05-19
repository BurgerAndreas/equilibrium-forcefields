import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
import copy
import os, sys, pathlib
import yaml
import json
import requests

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder, acclabels, timelabels




def main(
    run_id: str, datasplit: str = "train", error_type="abs", ymax=None, logscale=False
):
    # https://github.com/wandb/wandb/issues/3966

    # "n_fsolver_steps_{_datasplit}"
    # artifact_name = f"{error_type}_fixed_point_error_traj_{datasplit}"
    artifact_name = f"n_fsolver_steps_{datasplit}"
    alias = "latest"

    api = wandb.Api()
    run = api.run(project + "/" + run_id)
    run_name = run.name
    mname = ''.join(e for e in run_name if e.isalnum())


    print("Downloading run history...")
    history = run.scan_history()
    print("Processing run history...")
    losses = [
        [row[artifact_name], row["_step"]]
        for row in history
        if artifact_name in row.keys()
    ]
    print(f" Losses found: {len(losses[0][0]) if len(losses) > 0 else None}")

    print(f"Filtering out None values...")
    losses_nonone = [[r, s] for r, s in losses if r is not None]
    print(f" Rows that were None: {len(losses) - len(losses_nonone)} / {len(losses)}")
    losses = losses_nonone

    print(f"Combining data into dataframe...")
    # only  want the last value
    nstep_list = losses[-1][0]
    print(f"nstep_list: {nstep_list}")

    df = pd.DataFrame(nstep_list)

    # plot: x=solver_step, y=error_type, hue=train_step
    sns.lineplot(
        data=df, x="solver_step", y=error_type, 
        hue="train_step"
    )
    plt.xlabel("Fixed-point solver step")
    plt.ylabel(f"Fixed-point error ({error_type})")
    if logscale:
        plt.yscale("log")
    if ymax is not None:
        # cant plot 0 on logscale
        # plt.ylim(1e-12, ymax)
        plt.ylim(top=ymax)
    # legend title
    plt.title(f"{run_name}")

    fname = f"{plotfolder}/nsteps_traj_{datasplit}_{mname}.png"
    plt.savefig(fname)
    print(f"Saved plot to {fname}")

    # close the plot
    plt.close()
    plt.gca().clear()
    plt.gcf().clear()


    """ Plot number of occurenes of each nstep """
    # nstep to int
    df["solver_step"] = df["solver_step"].astype(int)
    # count occurences
    nstep_counts = df.groupby("solver_step").count()
    nstep_counts = nstep_counts.reset_index()
    nstep_counts.columns = ["solver_step", "count"]
    # plot
    sns.barplot(data=nstep_counts, x="solver_step", y="count")
    plt.xlabel("Fixed-point solver step")
    plt.ylabel("Number of occurences")
    plt.title(f"{run_name}")
    fname = f"{plotfolder}/nsteps_count_{datasplit}_{mname}.png"
    plt.savefig(fname)
    print(f"Saved plot to {fname}")

if __name__ == "__main__":

    # 
    run_id = "8uuq632s"
    main(run_id, error_type="abs", datasplit="train", logscale=True)