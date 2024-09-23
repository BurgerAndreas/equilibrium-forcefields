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

from deq2ff.plotting.style import (
    set_seaborn_style,
    PALETTE,
    entity,
    projectmd,
    plotfolder,
    set_style_after,
)


"""
Careful: results are stochastic since wandb returns randomly sampled datapoints from history
"""

def plot_fp_error_traj(
    run_id: str,
    datasplit: str = "train",
    error_type="abs",
    xmax=None,
    ymax=None,
    logscale=False,
    save_plot=False,
):
    # https://github.com/wandb/wandb/issues/3966

    artifact_name = f"{error_type}_fixed_point_error_traj_{datasplit}"
    alias = "latest"

    api = wandb.Api()
    run = api.run(projectmd + "/" + run_id)
    run_name = run.name
    print("\nrun_id:", run_id)
    print("name:", run.name)

    # metrics_dataframe = run.history()
    mname = "".join(e for e in run_name if e.isalnum())
    csvname = f"{plotfolder}/{run.id}_{error_type}_fixed_point_error_traj_full_{datasplit}.csv"
    # try to load from csv
    # try:
    #     df = pd.read_csv(csvname)
    #     print(f"Loaded from csv: {csvname}")

    # except FileNotFoundError:
    print("Downloading run history...")
    df = run.history()
    print(f"Processing run history (length {len(df)})...")
    # print(f"df: \n{df}")

    # abs_fixed_point_error_train
    # abs_fixed_point_error_traj_train
    # print("abs_fixed_point_error_traj_train\n", df["abs_fixed_point_error_traj_train"])

    # make step same as index
    df["step"] = df.index
    df["step"] = df["step"].astype(int)
    # drop first row
    df = df.drop(df.index[0])

    # take one epoch first rows
    maxsteps = 950 // 4
    df = df.head(maxsteps)

    # plot each row of the df as a line
    df = df.rename(columns={"abs_fixed_point_error_traj_train": "abs"})
    df = df.rename(columns={"rel_fixed_point_error_traj_train": "rel"})

    # drop all rows where rel is NaN
    df = df[df["rel"].notna()]
    print("len(df) notna:", len(df))
    print(df[["abs"]])

    # df = df["abs"]
    dfs = []
    # loop over the rows of the dataframe
    for i in range(len(df)):
        # get the row
        row = df.iloc[i]
        # create a new dataframe from the list
        print('row["abs"]:', row["abs"])
        dfi = pd.DataFrame(
            {
                "abs": row["abs"],
                "solver_step": range(len(row["abs"])),
                # "_step": i,
                "_step": [row["step"].item()] * len(row["abs"]),  # Expand scalar to match length
            }
        )
        # append the new dataframe to the list
        dfs.append(dfi)

    # concatenate the list of dataframes
    df = pd.concat(dfs)

    set_seaborn_style()
    fig, ax = plt.subplots()

    sns.lineplot(data=df, y="abs", x="solver_step", hue="_step", ax=ax)

    plt.xlabel("Fixed-point solver step")
    plt.ylabel(f"Fixed-point error ({error_type})")
    if logscale:
        plt.yscale("log")
    if ymax is not None:
        # cant plot 0 on logscale
        # plt.ylim(1e-12, ymax)
        plt.ylim(top=ymax)
    if xmax is not None:
        plt.xlim(left=0, right=xmax)
    # legend title
    # plt.title(f"{run_name}")
    plt.title(f"Fixed-Point Trace over Training")

    set_style_after(ax, fs=10)

    # legend outside the plot on the right
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    # ax.get_legend().get_frame().set_linewidth(0.0)

    plt.tight_layout()

    if save_plot:
        fname = f"{plotfolder}/fixed_point_error_traj_{datasplit}_{error_type}_{run_id.split('/')[-1]}_{mname}.png"
        plt.savefig(fname)
        print(f"Saved plot to \n {fname}")

    return fig, ax
    # close the plot
    # plt.cla()
    # plt.clf()
    # plt.close()


if __name__ == "__main__":

    # ----------------- E2 paper -----------------
    # DEQE2 fpcof inf-fptrace numlayers-2 iew27536
    # run_id = "iew27536"
    # main(run_id, error_type="abs", datasplit="train", logscale=True)
    run_id = "iew27536"
    plot_fp_error_traj(
        run_id, error_type="abs", datasplit="train", logscale=True, save_plot=True
    )
