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

""" Options """
filter_eval_batch_size = 4 # 1 or 4
filter_fpreuseftol = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
target = "aspirin" # aspirin, all
time_metric = "time_forward_per_batch_test" + "_lowest" # time_test, time_forward_per_batch_test, time_forward_total_test
acc_metric = "test_f_mae" + "_lowest" # test_f_mae_lowest, test_f_mae, test_e_mae_lowest, test_e_mae, best_test_f_mae, best_test_e_mae
layers_deq = [1, 2]
layers_equi = [1, 4, 8]
runs_with_dropout = False
# hosts = ["tacozoid11", "tacozoid10", "andreasb-lenovo"]
hosts, hostname = ["tacozoid11", "tacozoid10"], "taco"
# hosts, hostname = ["andreasb-lenovo"], "bahen"

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
    print(f"Found {len(run_ids)} runs with tag 'inference_speed'")

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
                "target": run.config["target"],
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

""" If FPReuse exists, use it """
# time_test_lowest should be lowest out of time_test and time_test_fpreuse
for m in time_metrics + acc_metrics:
    df[f"{m}_lowest"] = df.apply(lambda x: min(x[m], x[f"{m}_fpreuse"]), axis=1)


print('\nFiltering for target:', target)
df = df[df["target"] == target]

df = df[df["eval_batch_size"] == filter_eval_batch_size]

# fpreuse_f_tol="_default" -> 1e-3
df["fpreuse_f_tol"] = df["fpreuse_f_tol"].apply(lambda x: 1e-3 if x == "_default" else x)

nans = ['NaN', pd.NA, None, float("inf"), np.nan]
df = df[df["fpreuse_f_tol"].isin(filter_fpreuseftol + nans)]

# for Equiformer only keep Layers=[1,4, 8]
# df = df[df["Layers"].isin(layers)]
df = df[
    (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ")) | (df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer"))
]
# isin(layers_deq) and Model=DEQ or isin(layers_equi) and Model=Equiformer
# df = df[(df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer")) | (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ"))]

print('\nAfter filtering:\n', df[["Model", "Layers", "epoch", "evaluate", acc_metric, time_metric, "fpreuse_f_tol"]])


################################################################################################################################
# PLOTS
################################################################################################################################

color_palette = sns.color_palette('muted')
color_equiformer = color_palette[0]
color_deq = color_palette[1]
model_to_color = {"Equiformer": color_equiformer, "DEQ": color_deq}
equiformer_first = True

df.sort_values(by=["Model", "Layers", "fpreuse_f_tol"], inplace=True, ascending=[not equiformer_first, True, True])

df_clustered = copy.deepcopy(df)
df_clustered["fpreuse_f_tol"] = df_clustered["fpreuse_f_tol"].apply(lambda x: 0.0 if np.isnan(x) else x)
# combine cols_to_keep into one
df_clustered["run_name"] = df_clustered.apply(lambda x: f"{x['Model']} {x['Layers']}", axis=1)
df_clustered["run_name"] = df_clustered.apply(
    lambda x: x['run_name'] + f" {x['fpreuse_f_tol']:.0e}" if x['fpreuse_f_tol'] != 0.0 else x['run_name'], axis=1
)
cols_to_keep = ["Model", "Layers", "fpreuse_f_tol"] + ["run_name"] 
df_mean = df_clustered.groupby(cols_to_keep).mean(numeric_only=True).reset_index()
df_std = df_clustered.groupby(cols_to_keep).std(numeric_only=True).reset_index()
print('\nAfter renaming:\n', df_clustered[["run_name", "Model", "Layers", acc_metric, time_metric, "fpreuse_f_tol"]])
print('\nAfter averaging:\n', df_mean[["run_name", acc_metric, time_metric, "fpreuse_f_tol"]])

for _avg in [True, False]:
    if _avg:
        # barplots make their own mean and error bars
        _df = df_clustered
    else:
        _df = df

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
    # ax.set_xticks(loc[::2]) # TODO: this is a hack, only show every second label
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

y = acc_metric
x = time_metric
colorstyle = "Model"
shapestyle = "Layers"
# https://stackoverflow.com/a/64403147/18361030
marks = ["o", "s", "^"]

set_seaborn_style()

fig, ax = plt.subplots()

df_mean.sort_values(by=["Model"], inplace=True, ascending=[not equiformer_first])
print('\nMean for acc vs speed:\n', df_mean[[x, y, colorstyle, shapestyle]])

# error bars on both axes
# sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)
sns.scatterplot(
    data=df_mean, x=x, y=y, hue=colorstyle, style=shapestyle, ax=ax, markers=marks[:len(list(df[shapestyle].unique()))], s=200, 
)
# draws error bars and lines
for i, m in enumerate(list(df["Model"].unique())):
    ax.errorbar(
        df_mean[df_mean["Model"] == m][x], 
        df_mean[df_mean["Model"] == m][y], 
        yerr=df_std[df_std["Model"] == m][y], 
        xerr=df_std[df_std["Model"] == m][x], 
        # fmt='o', 
        # fmt='none', # no line
        lw=2,
        # color='black', 
        color=model_to_color[m],
        capsize=8,
        elinewidth=3,
        capthick=3,
        # legend=False,
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