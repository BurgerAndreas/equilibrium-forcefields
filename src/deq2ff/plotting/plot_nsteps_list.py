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

from deq2ff.plotting.style import (
    set_seaborn_style,
    PALETTE,
    entity,
    projectmd,
    plotfolder,
    human_labels,
    reset_plot_styles,
    set_style_after,
    myrc,
)


def plot_nsteps_list(
    run_id: str, ymax=None, xmax=None, logscale=False,
    as_perecent=False, show=False, palette=None,
    ):
    # https://github.com/wandb/wandb/issues/3966

    api = wandb.Api()
    run = api.run(f"{projectmd}/{run_id}")
    n_fsolver_steps_test = run.summary["n_fsolver_steps_test"]
    n_fsolver_steps_test_fpreuse = run.summary["n_fsolver_steps_test_fpreuse"]

    print("\nPlotting run:", run.name)

    # """ Plot number of occurenes of each nstep """
    # for fpreuse in [True, False]:
    #     if fpreuse:
    #         df = pd.DataFrame(n_fsolver_steps_test_fpreuse)
    #         # print(f"fpreuse={fpreuse}: {len(df)} steps logged")
    #     else:
    #         df = pd.DataFrame(n_fsolver_steps_test)
    #         # print(f"fpreuse={fpreuse}: {len(df)} steps logged")

    #     # name columns
    #     df.columns = ["nstep"]

    #     """ Plot 1 """
    #     # df["nstep"] = df["nstep"].round(0).astype(int)
    #     # nstep_counts = df.value_counts(["nstep"])
    #     # nstep_counts = nstep_counts.reset_index()
    #     # nstep_counts.columns = ["nstep", "count"]
    #     # nstep_counts.sort_values(by="nstep", inplace=True)
    #     # nstep_counts.reset_index(drop=True, inplace=True)
    #     # print('\nnstep_counts:\n', nstep_counts)

    #     # sns.scatterplot(data=nstep_counts, x="nstep", y="count")
    #     # sns.displot(data=nstep_counts, x="nstep", y="count", binwidth=1)
    #     # sns.displot(data=nstep_counts, x="nstep", y="count", kind="kde", fill=True)
    #     # sns.histplot(data=nstep_counts, x="nstep", y="count")
    #     # sns.barplot(data=nstep_counts, x="nstep", y="count")

    #     set_seaborn_style()
    #     df["nstep"] = df["nstep"].round(0).astype(int)

    #     # plot: x=solver_step, y=error_type, hue=train_step
    #     # A special case for the bar plot is when you want to show the number of observations in each category
    #     # rather than computing a statistic for a second variable.
    #     # This is similar to a histogram over a categorical, rather than quantitative, variable.
    #     # In seaborn, it’s easy to do so with the countplot()
    #     sns.countplot(
    #         data=df,
    #         x="nstep",
    #         dodge=False,
    #         # hue="nstep", palette="viridis",
    #     )

    #     plt.gcf().set_size_inches(8, 6)

    #     plt.xlabel("Fixed-point solver steps")
    #     plt.ylabel(f"Number of occurences")
    #     if logscale:
    #         plt.yscale("log")
    #     if ymax is not None:
    #         # cant plot 0 on logscale
    #         # plt.ylim(1e-12, ymax)
    #         plt.ylim(top=ymax)
    #     # legend title
    #     plt.title(f"Solver Steps {'with' if fpreuse else 'without'} FPreuse")

    #     plt.tight_layout()

    #     mname = run.name  # wandb.run.name # args.checkpoint_wandb_name
    #     # remove special characters
    #     mname = "".join(e for e in mname if e.isalnum())
    #     fname = (
    #         f"{plotfolder}/nsteps_traj"
    #         + f"{'_fpreuse' if fpreuse else ''}"
    #         + f"_{mname}.png"
    #     )
    #     plt.savefig(fname)
    #     print(f"Saved plot to \n {fname}")

    #     # close the plot
    #     plt.close()
    #     plt.gca().clear()
    #     plt.gcf().clear()
    #     plt.clf()

    """ Plot number of occurenes of each nstep with and without fpreuse in the same plot """
    df = pd.DataFrame(n_fsolver_steps_test)
    df.columns = ["nstep"]
    df_fp = pd.DataFrame(n_fsolver_steps_test_fpreuse)
    df_fp.columns = ["nstep"]
    # print(f"fpreuse=False: {len(df)} steps logged")
    # print(f"fpreuse=True: {len(df_fp)} steps logged")

    if len(df) > 1.5 * len(df_fp):
        # remove every second entry, starting with the 0th entry
        df = df.iloc[::2]
        # print(f"removed every second entry, len(df)={len(df)}")

    num_samples = len(df)

    df["class"] = "No fixed-point reuse"
    df_fp["class"] = "Fixed-point reuse"

    # concatenate the two dataframes
    _df = pd.concat([df, df_fp], ignore_index=True)

    _df["nstep"] = _df["nstep"].round(0).astype(int)

    set_seaborn_style()

    g = sns.catplot(
        data=_df,
        x="nstep",
        hue="class",
        # “strip”, “swarm”, “box”, “violin”, “boxen”, “point”, “bar”, or “count”
        kind="count",
        # palette="pastel",
        palette=PALETTE if palette is None else palette,
        # figsize = (8, 6), # Width, height
        # height=6, aspect=1.,
        # edgecolor=".6",
        # legend=False,
        # height=6, aspect=1.5,
        native_scale=True,
        # width=0.8,
        gap=0.0,
        dodge=False,
        # rc=myrc,
    )

    # set size to (8,6)
    plt.gcf().set_size_inches(8, 6)
    fig = g.figure
    fig.set_size_inches(8, 6)

    # add small xticks
    maxx = plt.gca().get_xlim()[1]
    # plt.xticks(np.arange(0, maxx, 2 if logscale else 1))
    if logscale or xmax is None or xmax > 10:
        plt.xticks(np.arange(0, maxx, 2))
    else:
        plt.xticks(np.arange(0, maxx, 1))

    # turn of grid
    plt.grid(False)
    plt.grid(which="major", axis="y", linestyle="-", linewidth="1.0", color="lightgray")

    # set_style_after(ax)
    # set runtime configuration (rc) parameters
    g.figure.set_linewidth(myrc["lines.linewidth"])

    # set font size of labels
    g.set_axis_labels(
        "Fixed-point solver steps", "Number of occurences", fontsize=myrc["font.size"]
    )
    # g.set_titles("Solver steps with and without fixed-point reuse", fontsize=myrc["font.size"])
    g.set_xticklabels(fontsize=myrc["font.size"])
    g.set_yticklabels(fontsize=myrc["font.size"])

    # g.facet_axis(0, 0).set_xlabel("Fixed-point solver steps")

    plt.xlabel("Fixed-Point Solver Steps")
    plt.ylabel(f"Number of Occurences")
    if logscale:
        plt.yscale("log")
    if ymax is not None:
        # cant plot 0 on logscale
        # plt.ylim(1e-12, ymax)
        plt.ylim(top=ymax)
    if xmax is not None:
        plt.xlim(right=xmax)
    # legend title
    plt.title(f"Solver Steps w/wo Fixed-point Reuse", fontsize=myrc["font.size"])

    if as_perecent:
        # turn count into percentage
        if logscale:
            g.set(ylabel="Percentage (% log scale)")
        else:
            g.set(ylabel="Percentage (%)")
        g.ax.set_ylim(0, num_samples)
        # replace yticks by dividing by len(df)
        yticks = g.ax.get_yticks()
        yticks = np.round(yticks / num_samples * 100, 1)
        yticks = [_y if (_y > 0 and _y < 1) else int(_y) for _y in yticks]
        g.ax.set_yticklabels(yticks)
        # g.ax.set_yticks(yticks / num_samples)
        # set ymax to 100

    # ax = set_style_after(g)

    # no legend title
    # plt.legend(title=None)
    legend = g._legend
    legend.set_title(None)
    # set to top right
    legend.set_bbox_to_anchor([1.0, 0.87])

    plt.tight_layout(pad=0.1)

    mname = run.name  # wandb.run.name # args.checkpoint_wandb_name
    # remove special characters
    mname = "".join(e for e in mname if e.isalnum())
    if logscale:
        mname += "_logscale"
    fname = f"{plotfolder}/nsteps_wo_fpreuse_{mname}.png"
    plt.savefig(fname)
    print(f"Saved plot to \n {fname}")

    if show:
        plt.show()
    plt.cla()
    plt.clf()
    plt.close()

def search_for_runs():
    # get all runs with tag 'inference_speed'
    api = wandb.Api()
    runs = api.runs(
        projectmd,
        {
            "tags": "inference",
            # "$or": [{"tags": "md17"}, {"tags": "md22"}, {"tags": "main2"}],
            # "state": "finished",
            # $or": [{"tags": "md17"}, {"tags": "main2"}, {"tags": "inference"}],
            # "$or": [{"state": "finished"}, {"state": "crashed"}],
        },
    )
    run_ids = [run.id for run in runs]
    print(f"Found {len(run_ids)} runs:")

    time_metrics = [
        "time_test",
        "time_forward_per_batch_test",
        "time_forward_total_test",
    ]
    acc_metrics = ["test_f_mae", "test_e_mae"]

    infos = []
    for run in runs:
        try:
            # # model.path_drop=0.05
            # if runs_with_dropout:
            #     if run.config["model"]["path_drop"] != 0.05:
            #         continue
            # else:
            #     if run.config["model"]["path_drop"] != 0.0:
            #         continue
            # if run.summary["epoch"] < 995:
            #     continue
            print(" ", run.name, "| run_id:", run.id, "| epoch:", run.summary["epoch"])
            info = {
                "run_id": run.id,
                "run_name": run.name,
                "seed": run.config["seed"],
                "num_layers": run.config["model"]["num_layers"],
                "model_is_deq": run.config["model_is_deq"],
                "Target": run.config["target"],
                "Params": run.summary["Model Parameters"],
                "PathDrop": run.config["model"]["path_drop"],
                "Alpha": run.config["model"]["alpha_drop"],
                # "epoch": run.summary["epoch"],
                "n_fsolver_steps_test": run.summary["n_fsolver_steps_test"],
                "n_fsolver_steps_test_fpreuse": run.summary[
                    "n_fsolver_steps_test_fpreuse"
                ],
            }
            # Plots: pick the smaller of test_fpreuse_f_mae and test_f_mae
            for _m in acc_metrics + time_metrics:
                if _m in run.summary:
                    info[_m] = run.summary[_m]
                if _m.replace("test", "test_fpreuse") in run.summary:
                    info[_m] = min(
                        run.summary[_m], run.summary[_m.replace("test", "test_fpreuse")]
                    )
            # if 'test_fpreuse_f_mae' in run.summary:
            #     info["test_f_mae"] = min(run.summary["test_f_mae"], run.summary["test_fpreuse_f_mae"])
            #     info["test_e_mae"] = min(run.summary["test_e_mae"], run.summary["test_fpreuse_e_mae"])
            optionals = ["epoch", "best_test_e_mae", "best_test_f_mae"]
            for _o in optionals:
                if _o in run.summary:
                    info[_o] = run.summary[_o]
        except KeyError as e:
            print(f"Skipping run {run.id} {run.name} because of KeyError: {e}")
            continue
        infos.append(info)

    return infos

if __name__ == "__main__":


    # run_id = "xpz4crad"
    # run_id = "o732ps0t"
    # plot_nsteps_list(run_id, logscale=False, xmax=15)
    # plot_nsteps_list(run_id, logscale=True)

    # pDEQs apt inf-bs1acc 
    run_id = "b321vc1w"
    plot_nsteps_list(run_id, logscale=False, xmax=9, as_perecent=True)
    # plot_nsteps_list(run_id, logscale=True)
    plot_nsteps_list(run_id, logscale=True, as_perecent=True)

    # fpiter
    run_id = "o16dbur0"
    plot_nsteps_list(run_id, logscale=False, xmax=9, as_perecent=True)
    plot_nsteps_list(run_id, logscale=True, as_perecent=True)
    
    # pDEQs ap ln-pre malonaldehyde
    # 7x83gn1c
    # # fpiter
    run_id = "7x83gn1c"
    plot_nsteps_list(run_id, logscale=False, xmax=9, as_perecent=True)
    plot_nsteps_list(run_id, logscale=True, as_perecent=True)
