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

from deq2ff.plotting.style import set_seaborn_style, PALETTE, entity, project, plotfolder, acclabels, timelabels, set_style_after, myrc



def main(
    run_id: str, ymax=None, xmax=None, logscale=False
):
    # https://github.com/wandb/wandb/issues/3966

    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    n_fsolver_steps_test = run.summary["n_fsolver_steps_test"]
    n_fsolver_steps_test_fpreuse = run.summary["n_fsolver_steps_test_fpreuse"]

    print('Plotting run:', run.name)

    for fpreuse in [True, False]:
        if fpreuse:
            df = pd.DataFrame(n_fsolver_steps_test_fpreuse)
        else:
            df = pd.DataFrame(n_fsolver_steps_test)

        # name columns
        df.columns = ["nstep"]

        """ Plot 1 """
        # df["nstep"] = df["nstep"].round(0).astype(int)
        # nstep_counts = df.value_counts(["nstep"])
        # nstep_counts = nstep_counts.reset_index()
        # nstep_counts.columns = ["nstep", "count"]
        # nstep_counts.sort_values(by="nstep", inplace=True)
        # nstep_counts.reset_index(drop=True, inplace=True)
        # print('\nnstep_counts:\n', nstep_counts)
        
        # sns.scatterplot(data=nstep_counts, x="nstep", y="count")
        # sns.displot(data=nstep_counts, x="nstep", y="count", binwidth=1)
        # sns.displot(data=nstep_counts, x="nstep", y="count", kind="kde", fill=True)
        # sns.histplot(data=nstep_counts, x="nstep", y="count")
        # sns.barplot(data=nstep_counts, x="nstep", y="count")

        """ Plot number of occurenes of each nstep """
        set_seaborn_style()
        df["nstep"] = df["nstep"].round(0).astype(int)

        # plot: x=solver_step, y=error_type, hue=train_step
        # A special case for the bar plot is when you want to show the number of observations in each category 
        # rather than computing a statistic for a second variable. 
        # This is similar to a histogram over a categorical, rather than quantitative, variable. 
        # In seaborn, it’s easy to do so with the countplot()
        sns.countplot(
            data=df, x="nstep", dodge=False,
            # hue="nstep", palette="viridis",
        )

        plt.gcf().set_size_inches(8, 6)

        plt.xlabel("Fixed-point solver steps")
        plt.ylabel(f"Number of occurences")
        if logscale:
            plt.yscale("log")
        if ymax is not None:
            # cant plot 0 on logscale
            # plt.ylim(1e-12, ymax)
            plt.ylim(top=ymax)
        # legend title
        plt.title(f"Solver steps {'with' if fpreuse else 'without'} fpreuse")

        plt.tight_layout()

        mname = run.name # wandb.run.name # args.checkpoint_wandb_name
        # remove special characters
        mname = ''.join(e for e in mname if e.isalnum())
        fname = f"{plotfolder}/nsteps_traj" + f"{'_fpreuse' if fpreuse else ''}" + f"_{mname}.png"
        plt.savefig(fname)
        print(f"Saved plot to \n {fname}")

        # close the plot
        plt.close()
        plt.gca().clear()
        plt.gcf().clear()
        plt.clf()

    """ Plot number of occurenes of each nstep with and without fpreuse in the same plot """
    df = pd.DataFrame(n_fsolver_steps_test)
    df.columns = ["nstep"]
    df_fp = pd.DataFrame(n_fsolver_steps_test_fpreuse)
    df_fp.columns = ["nstep"]

    df["class"] = "No fixed-point reuse"
    df_fp["class"] = "Fixed-point reuse"

    
    # concatenate the two dataframes
    _df = pd.concat([df, df_fp], ignore_index=True)
    
    _df["nstep"] = _df["nstep"].round(0).astype(int)

    set_seaborn_style()

    g = sns.catplot(
        data=_df, x="nstep", hue="class", 
        kind="count",
        # palette="pastel", 
        palette=PALETTE,
        # figsize = (8, 6), # Width, height
        # height=6, aspect=1.,
        # edgecolor=".6",
        # legend=False,
        # height=6, aspect=1.5,
        native_scale=True,
        # width=0.8,
        gap=0.0,
        dodge=False,
        rc=myrc,
    )

    # set size to (8,6)
    plt.gcf().set_size_inches(8, 6)
    fig = g.figure
    fig.set_size_inches(8, 6)

    # add small xticks
    maxx = plt.gca().get_xlim()[1]
    # plt.xticks(np.arange(0, maxx, 2 if logscale else 1))
    plt.xticks(np.arange(0, maxx, 2))

    # turn of grid
    plt.grid(False)
    plt.grid(which='major', axis='y', linestyle='-', linewidth='1.0', color='lightgray')

    # set_style_after(ax)

    plt.xlabel("Fixed-point solver steps")
    plt.ylabel(f"Number of occurences")
    if logscale:
        plt.yscale("log")
    if ymax is not None:
        # cant plot 0 on logscale
        # plt.ylim(1e-12, ymax)
        plt.ylim(top=ymax)
    if xmax is not None:
        plt.xlim(right=xmax)
    # legend title
    plt.title(f"Solver steps w/o fixed-point reuse")

    # no legend title
    # plt.legend(title=None)
    legend = g._legend
    legend.set_title(None)
    # set to top right
    legend.set_bbox_to_anchor([0.95, 0.82])

    plt.tight_layout()

    mname = run.name # wandb.run.name # args.checkpoint_wandb_name
    # remove special characters
    mname = ''.join(e for e in mname if e.isalnum())
    if logscale:
        mname += "_logscale"
    fname = f"{plotfolder}/nsteps_wo_fpreuse_{mname}.png"
    plt.savefig(fname)
    print(f"Saved plot to \n {fname}")




if __name__ == "__main__":

    # get all runs with tag 'inference_speed'
    api = wandb.Api()
    runs = api.runs(
        project, 
        {
            "tags": "inference", 
            # "$or": [{"tags": "md17"}, {"tags": "md22"}, {"tags": "main2"}],
            # "state": "finished",
            # $or": [{"tags": "md17"}, {"tags": "main2"}, {"tags": "inference"}],
            # "$or": [{"state": "finished"}, {"state": "crashed"}],
        }
    )
    run_ids = [run.id for run in runs]
    print(f"Found {len(run_ids)} runs:")

    time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]
    acc_metrics = ["test_f_mae", "test_e_mae"]

    infos = []
    for run in runs:
        try:
            # # model.drop_path_rate=0.05
            # if runs_with_dropout:
            #     if run.config["model"]["drop_path_rate"] != 0.05:
            #         continue
            # else:
            #     if run.config["model"]["drop_path_rate"] != 0.0:
            #         continue
            # if run.summary["epoch"] < 995:
            #     continue
            print(' ', run.name, '| run_id:', run.id, '| epoch:', run.summary["epoch"])
            info = {
                "run_id": run.id,
                "run_name": run.name,
                "seed": run.config["seed"],
                "num_layers": run.config["model"]["num_layers"],
                "model_is_deq": run.config["model_is_deq"],
                "Target": run.config["target"],
                "Params": run.summary["Model Parameters"],
                "PathDrop": run.config["model"]["drop_path_rate"],
                "Alpha": run.config["model"]["alpha_drop"],
                # "epoch": run.summary["epoch"],
                "n_fsolver_steps_test": run.summary["n_fsolver_steps_test"],
                "n_fsolver_steps_test_fpreuse": run.summary["n_fsolver_steps_test_fpreuse"],
            }
            # Plots: pick the smaller of test_fpreuse_f_mae and test_f_mae
            for _m in acc_metrics + time_metrics:
                if _m in run.summary:
                    info[_m] = run.summary[_m]
                if _m.replace("test", "test_fpreuse") in run.summary:
                    info[_m] = min(run.summary[_m], run.summary[_m.replace("test", "test_fpreuse")])
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

    # 
    # run_id = "xpz4crad"
    run_id = "o732ps0t"
    main(run_id, logscale=False, xmax=15)
    main(run_id, logscale=True)