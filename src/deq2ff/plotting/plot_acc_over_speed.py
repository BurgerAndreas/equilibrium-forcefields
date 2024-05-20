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

nans = ['NaN', pd.NA, None, float("inf"), np.nan]

def plot_speed_acc_target(dfc):

    color_palette = sns.color_palette('muted')
    color_equiformer = color_palette[0]
    color_deq = color_palette[1]
    model_to_color = {"Equiformer": color_equiformer, "DEQ": color_deq}
    equiformer_first = True

    dfc.sort_values(by=["Model", "Layers", "fpreuse_f_tol"], inplace=True, ascending=[not equiformer_first, True, True])

    df_clustered = copy.deepcopy(dfc)
    df_clustered["fpreuse_f_tol"] = df_clustered["fpreuse_f_tol"].apply(lambda x: 0.0 if np.isnan(x) else x)
    # combine cols_to_keep into one
    df_clustered["run_name"] = df_clustered.apply(lambda x: f"{x['Model']} {x['Layers']}", axis=1)
    df_clustered["run_name"] = df_clustered.apply(
        lambda x: x['run_name'] + f" {x['fpreuse_f_tol']:.0e}" if x['fpreuse_f_tol'] != 0.0 else x['run_name'], axis=1
    )
    cols_to_keep = ["Model", "Layers", "fpreuse_f_tol"] + ["run_name"] 
    df_mean = df_clustered.groupby(cols_to_keep).mean(numeric_only=True).reset_index()
    df_std = df_clustered.groupby(cols_to_keep).std(numeric_only=True).reset_index()
    # print('\nAfter renaming:\n', df_clustered[["run_name", "Model", "Layers", acc_metric, time_metric, "fpreuse_f_tol"]])
    # print('\nAfter averaging:\n', df_mean[["run_name", acc_metric, time_metric, "fpreuse_f_tol"]])

    for _avg in [True, False]:
        if _avg:
            # barplots make their own mean and error bars
            _df = df_clustered
        else:
            _df = dfc

        sns.set_palette(color_palette)

        """ Barchart of inference time """
        y = time_metric
        x = "run_name"
        color = "Model"

        # plot
        # set_seaborn_style(figsize=(10, 5))
        # sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=_df, x=x, y=y, hue=color, ax=ax)

        # write values on top of bars
        for p in ax.patches:
            # do not write 0.00
            if p.get_height() == 0:
                continue
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

        # make labels vertical
        # plt.xticks(rotation=90)

        loc, labels = plt.xticks()
        # ax.set_xticks(loc[::2]) # this is a hack, only show every second label
        ax.set_xticks(loc) 
        ax.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize=8)

        # labels
        ax.set_xlabel("") # "Run name"
        ax.set_ylabel(timelabels[y.replace("_lowest", "")])

        plt.tight_layout()

        # save
        name = f"speed2{'-avg' if _avg else ''}-bs{filter_eval_batch_size}-{time_metric.replace('_lowest', '')}"
        plt.savefig(f"{plotfolder}/{name}.png")
        print(f"\nSaved plot to \n {plotfolder}/{name}.png")


        """ Barchart of accuracy """
        y = acc_metric
        x = "run_name"
        color = "Model"

        # plot
        # set_seaborn_style(figsize=(10, 5))
        # sns.set_style("whitegrid")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=_df, x=x, y=y, hue=color, ax=ax)

        # write values on top of bars
        for p in ax.patches:
            # do not write 0.00
            if p.get_height() == 0:
                continue
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

        # make labels vertical
        # plt.xticks(rotation=90)

        loc, labels = plt.xticks()
        ax.set_xticks(loc)
        # UserWarning: set_ticklabels() should only be used with a fixed number of ticks, i.e. after set_ticks() or using a FixedLocator.
        ax.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize=8)

        # labels
        ax.set_xlabel("") # "Run name"
        ax.set_ylabel(acclabels[y.replace("_lowest", "")])

        plt.tight_layout()

        # save
        name = f"acc2{'-avg' if _avg else ''}-bs{filter_eval_batch_size}-{acc_metric.replace('_lowest', '')}"
        plt.savefig(f"{plotfolder}/{name}.png")
        print(f"\nSaved plot to \n {plotfolder}/{name}.png")

    """ Plot accuracy over inference time"""

    # only plot one point TODO
    # df = df[df["fpreuse_f_tol"].isin([1e0] + nans)]
    # select the lower
    m = "test_f_mae"
    mfp = m.replace('test', 'test_fpreuse')
    df[f"{m}_lowest"] = df.apply(lambda x: min(x[m], x[mfp]), axis=1)

    y = acc_metric
    x = time_metric
    colorstyle = "Model"
    shapestyle = "Layers"
    # https://stackoverflow.com/a/64403147/18361030
    # marks = ["o", "s", "^"]
    marks = ["o", "X", "^", "P"]
    # marks = ["o", "P", "^"]

    set_seaborn_style()

    fig, ax = plt.subplots()

    df_mean.sort_values(by=["Model"], inplace=True, ascending=[not equiformer_first])
    print('\nMean for acc vs speed:\n', df_mean[[x, y, colorstyle, shapestyle]])

    # error bars on both axes
    # shades of blue
    blues = sns.color_palette("Blues", n_colors=3)
    # oranges
    oranges = sns.color_palette("Reds", n_colors=6)
    model_to_colors = {"Equiformer": blues, "DEQ": oranges}
    _i = 0
    # draws error bars and lines
    for i, m in enumerate(list(dfc["Model"].unique())):
        for _l, l in enumerate(list(dfc[shapestyle].unique())):
            _mean = copy.deepcopy(df_mean)
            _mean = _mean[_mean["Model"] == m]
            _mean = _mean[_mean[shapestyle] == l]
            _std = copy.deepcopy(df_std)
            _std = _std[_std["Model"] == m]
            _std = _std[_std[shapestyle] == l]
            ax.errorbar(
                _mean[x], 
                _mean[y], 
                xerr=_std[x], 
                yerr=_std[y], 
                # fmt='o', 
                # fmt='none', # no line
                lw=2,
                # color='black', 
                color=model_to_color[m],
                # color=model_to_colors[m][_l],
                capsize=5,
                elinewidth=2,
                capthick=2,
                # legend=False,
                alpha=0.5,
            )
            _i += 1

    # sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)
    sns.scatterplot(
        data=df_mean, x=x, y=y, hue=colorstyle, style=shapestyle, ax=ax, markers=marks[:len(list(dfc[shapestyle].unique()))], s=200, 
    )

    # labels
    ax.set_xlabel(timelabels[x.replace("_lowest", "")])
    ax.set_ylabel(r"Force MAE [meV/$\AA$]")
    ax.set_title("Inference speed vs. accuracy")

    # ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

    plt.tight_layout(pad=0.1)

    # save
    name = f"acc_over_inferencetime" + f"-bs{filter_eval_batch_size}-{time_metric}"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to \n {plotfolder}/{name}.png")


""" Options """
filter_eval_batch_size = 4 # 1 or 4
filter_fpreuseftol = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
Target = "aspirin" # aspirin, all
time_metric = "time_forward_per_batch_test" + "_lowest" # time_test, time_forward_per_batch_test, time_forward_total_test
acc_metric = "test_f_mae" + "_lowest" # test_f_mae_lowest, test_f_mae, test_e_mae_lowest, test_e_mae, best_test_f_mae, best_test_e_mae
layers_deq = [1, 2]
layers_equi = [1, 4, 8]
runs_with_dropout = True
# hosts = ["tacozoid11", "tacozoid10", "andreasb-lenovo"]
# hosts, hostname = ["tacozoid11", "tacozoid10"], "taco"
hosts, hostname = ["andreasb-lenovo"], "bahen"

# download data or load from file
download_data = False

# choose from
eval_batch_sizes = [1, 4]
time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]
acc_metrics = ["test_f_mae", "test_e_mae"] # + ["best_test_f_mae", "best_test_e_mae"]
acclabels.update({f"{k}_lowest": v for k, v in acclabels.items()})

""" Load data """
if download_data:
    # get all runs with tag 'inference_speed'
    api = wandb.Api()
    runs = api.runs(
        project, 
        {
            "tags": "inference", "state": "finished",
            # $or": [{"tags": "md17"}, {"tags": "main2"}, {"tags": "inference"}],
            # "state": "finished",
            # "$or": [{"state": "finished"}, {"state": "crashed"}],
        }
    )
    run_ids = [run.id for run in runs]
    print(f"Found {len(run_ids)} runs:")

    infos_acc = []
    for run in runs:
        # run = api.run(project + "/" + run_id)
        print(' ', run.name)
        # meta = json.load(run.file("wandb-metadata.json").download(replace=True))
        # meta["host"]
        # host = requests.get(run.file("wandb-metadata.json").url).json()['host']
        try:
            # model.drop_path_rate=0.05
            if runs_with_dropout:
                if run.config["model"]["drop_path_rate"] != 0.05:
                    continue
            else:
                if run.config["model"]["drop_path_rate"] != 0.0:
                    continue
            host = requests.get(run.file("wandb-metadata.json").url).json()['host']
            if host not in hosts:
                print(f"Skipping run {run.id} {run.name} because of host={host}")
                continue
            info = {
                "run_id": run.id,
                "run_name": run.name,
                "eval_batch_size": run.config["eval_batch_size"],
                "Target": run.config["target"],
                "Parameters": run.summary["Model Parameters"],
                "seed": run.config["seed"],
                "num_layers": run.config["model"]["num_layers"],
                "model_is_deq": run.config["model_is_deq"],
                "evaluate": run.config["evaluate"],
            }
            summary_keys = time_metrics + acc_metrics
            for key in summary_keys:
                info[key] = run.summary[key]
        except KeyError as e:
            print(f"Skipping run {run.id} {run.name} because of KeyError: {e}")
            continue
        # Plots: pick the smaller of test_fpreuse_f_mae and test_f_mae
        if 'test_fpreuse_f_mae' in run.summary:
            info["test_fpreuse_f_mae"] = run.summary["test_fpreuse_f_mae"]
            info["test_fpreuse_e_mae"] = run.summary["test_fpreuse_e_mae"]
            # info["test_f_mae"] = min(run.summary["test_f_mae"], run.summary["test_fpreuse_f_mae"])
            # info["test_e_mae"] = min(run.summary["test_e_mae"], run.summary["test_fpreuse_e_mae"])
            # info["best_test_fpreuse_f_mae"] = run.summary["best_test_fpreuse_f_mae"] # not recorded
            # info["best_test_fpreuse_e_mae"] = run.summary["best_test_fpreuse_e_mae"]
        else:
            info["test_fpreuse_f_mae"] = float("inf")
            info["test_fpreuse_e_mae"] = float("inf")
        optional_summary_keys = [_m + "_fpreuse" for _m in time_metrics] + [_m + "_fpreuse" for _m in acc_metrics]
        for key in optional_summary_keys:
            if key in run.summary:
                info[key] = run.summary[key]
            else:
                info[key] = float("inf")
        if "deq_kwargs_test" in run.config:
            info["fpreuse_f_tol"] = run.config["deq_kwargs_test"]["fpreuse_f_tol"]
        # evaluate does not have best_test_e_mae and best_test_f_mae
        try:
            info["best_test_e_mae"] = run.summary["best_test_e_mae"]
            info["best_test_f_mae"] = run.summary["best_test_f_mae"]
        except KeyError as e:
            info["best_test_e_mae"] = run.summary["test_e_mae"]
            info["best_test_f_mae"] = run.summary["test_f_mae"]
        if "epoch" in run.summary:
            info["epoch"] = run.summary["epoch"]
        if "_step" in run.summary:
            info["_step"] = run.summary["_step"]

        infos_acc.append(info)

    df = pd.DataFrame(infos_acc)

    # save dataframe
    df.to_csv(f"{plotfolder}/acc_over_speed_2-{hostname}.csv", index=False)

else:
    # load dataframe
    df = pd.read_csv(f"{plotfolder}/acc_over_speed_2-{hostname}.csv")

"""Rename columns"""
# rename 'model_is_deq' to 'Model'
# true -> DEQ, false -> Equiformer
df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
# rename 'model_is_deq' to 'Model'
df = df.rename(columns={"model_is_deq": "Model"})

# rename for prettier legends
df = df.rename(columns={"num_layers": "Layers"})

# cast test_fpreuse_f_mae to float
df["test_fpreuse_f_mae"] = df["test_fpreuse_f_mae"].astype(float)

""" If FPReuse exists, use it """
# time_test_lowest should be lowest out of time_test and time_test_fpreuse
# for m in time_metrics + acc_metrics:
for m in time_metrics:
    mfp = m.replace('test', 'test_fpreuse')
    df[f"{m}_lowest"] = df.apply(lambda x: min(x[m], x[mfp]), axis=1)
for m in acc_metrics:
    mfp = m.replace('test', 'test_fpreuse')
    df[f"{m}_lowest"] = df.apply(lambda x: x[mfp] if x[mfp] < 1000.0 else  x[m], axis=1)


df = df[df["eval_batch_size"] == filter_eval_batch_size]

# fpreuse_f_tol="_default" -> 1e-3
df["fpreuse_f_tol"] = df["fpreuse_f_tol"].apply(lambda x: 1e-3 if x == "_default" else x)

df = df[df["fpreuse_f_tol"].isin(filter_fpreuseftol + nans)]

# fpreuse_f_tol: replace nans with 0
# df["fpreuse_f_tol"] = df["fpreuse_f_tol"].apply(lambda x: 0.0 if np.isnan(x) else x)

# for Equiformer only keep Layers=[1,4, 8]
# df = df[df["Layers"].isin(layers)]
df = df[
    (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ")) | (df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer"))
]
# isin(layers_deq) and Model=DEQ or isin(layers_equi) and Model=Equiformer
# df = df[(df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer")) | (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ"))]

# print('\nFiltering for Target:', _Target)
df = df[df["Target"] == Target]

print('\nAfter filtering:\n', df[["Model", "Layers", "test_f_mae_lowest", "test_f_mae", "test_fpreuse_f_mae", "fpreuse_f_tol"]])


################################################################################################################################
# PLOTS
################################################################################################################################


plot_speed_acc_target(copy.deepcopy(df))
