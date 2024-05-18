import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
import copy
import os, sys, pathlib
import yaml

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder

""" Options """
filter_eval_batch_size = 4 # 1 or 4
filter_fpreuseftol = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5] + ['NaN', pd.NA, None, float("inf"), np.nan]
time_metric = "time_forward_per_batch_test" + "_lowest" # time_test, time_forward_per_batch_test, time_forward_total_test
target = "aspirin" # aspirin, all
acc_metric = "test_f_mae" # test_f_mae, test_e_mae, best_test_f_mae, best_test_e_mae
layers_deq = [1, 2]
layers_equi = [1, 4, 8]
runs_with_dropout = False

# choose from
eval_batch_sizes = [1, 4]
time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]
timelabels = {
    "time_test": "Test time for 1000 samples [s]",
    "time_forward_per_batch_test": "Forward time per batch [s]",
    "time_forward_total_test": "Total forward time for 1000 samples [s]",
}
acc_metrics = ["best_test_f_mae", "test_f_mae", "best_test_e_mae", "test_e_mae"]
acclabels = {
    "best_test_f_mae": "Best force MAE",
    "test_f_mae": "Force MAE",
    "best_test_e_mae": "Best energy MAE",
    "test_e_mae": "Energy MAE",
}

""" Load data """
# get all runs with tag 'inference_speed'
api = wandb.Api()
runs = api.runs(project, {"tags": "inference_speed", "state": "finished"})
run_ids = [run.id for run in runs]
print(f"Found {len(run_ids)} runs with tag 'inference_speed'")

# get runs with accuracy
runs_acc = api.runs(
    project, 
    filters={
        "$or": [{"tags": "md17"}, {"tags": "depth"}, {"tags": "inference_acc"}],
        "state": "finished"
    }
)
# runs_acc = api.runs(project, {"tags": "md17"})
# run_ids_acc = [run.id for run in runs_acc]
# #
# runs_acc = api.runs(project, {"tags": "depth"})
# run_ids_acc += [run.id for run in runs_acc]
# #
# runs_acc = api.runs(project, {"tags": "inference_acc"})
# run_ids_acc += [run.id for run in runs_acc]
# print(f"Found {len(run_ids_acc)} runs with tags 'md17' or 'depth'")


# get accuracy runs
print('\nAccuracy runs')
infos_acc = []
for run in runs_acc:
    # run = api.run(project + "/" + run_id)
    print(' ', run.name)
    try:
        # model.drop_path_rate=0.05
        if runs_with_dropout:
            if run.config["model"]["drop_path_rate"] != 0.05:
                continue
        else:
            if run.config["model"]["drop_path_rate"] != 0.0:
                continue
        info = {
            "run_id_acc": run.id,
            "run_name": run.name,
            "target": run.config["target"],
            "seed": run.config["seed"],
            "num_layers": run.config["model"]["num_layers"],
            "model_is_deq": run.config["model_is_deq"],
            # accuracy metrics
            "test_e_mae": run.summary["test_e_mae"],
            "test_f_mae": run.summary["test_f_mae"],
        }
        # Plots: pick the smaller of test_fpreuse_f_mae and test_f_mae
        if 'test_fpreuse_f_mae' in run.summary:
            info["test_f_mae"] = min(run.summary["test_f_mae"], run.summary["test_fpreuse_f_mae"])
            info["test_e_mae"] = min(run.summary["test_e_mae"], run.summary["test_epreuse_e_mae"])
    except KeyError as e:
        print(f"Skipping run {run.id} {run.name} because of KeyError: {e}")
        continue
    if "deq_kwargs_test" in run.config:
        info["fpreuse_f_tol"] = run.config["deq_kwargs_test"]["fpreuse_f_tol"]
    # evaluate does not have best_test_e_mae and best_test_f_mae
    try:
        info["best_test_e_mae"] = run.summary["best_test_e_mae"]
        info["best_test_f_mae"] = run.summary["best_test_f_mae"]
    except KeyError as e:
        info["best_test_e_mae"] = run.summary["test_e_mae"]
        info["best_test_f_mae"] = run.summary["test_f_mae"]
    infos_acc.append(info)

# TODO
# currently we select the one matching run by name, which includes the seed
# Option 1: get the same seeds for accuracy and speed runs
# option 2: create to separate df for accuracy and speed, average separately, then merge
df_acc = pd.DataFrame(infos_acc)
# filter for target
df_acc = df_acc[df_acc["target"] == target]
print('\nAccuracy runs:\n', df_acc[["run_name", "fpreuse_f_tol", acc_metric]])

print('\nSpeed runs')
optional_summary_keys = [_m + "_fpreuse" for _m in time_metrics]
infos = []
for run in runs:
    # model.drop_path_rate=0.05
    if runs_with_dropout:
        if run.config["model"]["drop_path_rate"] != 0.05:
            continue
    else:
        if run.config["model"]["drop_path_rate"] != 0.0:
            continue
    # run = api.run(project + "/" + run_id)
    # api = wandb.Api()
    print(' ', run.name)
    # print('run_config', yaml.dump(run.config))
    # exit()
    try:
        info = {
            "run_id_speed": run.id,
            "run_name": run.name,
            # "config": run.config,
            # "summary": run.summary,
            "seed": run.config["seed"],
            "num_layers": run.config["model"]["num_layers"],
            "model_is_deq": run.config["model_is_deq"],
            "eval_batch_size": run.config["eval_batch_size"],
            "target": run.config["target"],
            # time metrics
            "time_test": run.summary["time_test"],
            "time_forward_per_batch_test": run.summary["time_forward_per_batch_test"],
            "time_forward_total_test": run.summary["time_forward_total_test"],
        }
    except KeyError as e:
        print(f"Skipping run {run.id} {run.name} because of KeyError: {e}")
        raise e
    for key in optional_summary_keys:
        if key in run.summary:
            info[key] = run.summary[key]
        else:
            info[key] = float("inf")
    if "deq_kwargs_test" in run.config:
        info["fpreuse_f_tol"] = run.config["deq_kwargs_test"]["fpreuse_f_tol"]
    
    # find corresponding accuracy run
    for info_acc in infos_acc:
        # if info_acc["seed"] == info["seed"] and info_acc["num_layers"] == info["num_layers"]:
        _run_name = info["run_name"]
        _run_name = _run_name.replace(" evalbatchsize-1", "").replace(" evalbatchsize-4", "")
        _run_name_acc = info_acc["run_name"]
        # aspirin is the default
        _run_name_acc = _run_name_acc.replace(" target-aspirin", "")
        if _run_name_acc == _run_name:
            # ensure num_layers, target, seed are the same
            assert info_acc["num_layers"] == info["num_layers"], f"num_layers: {info_acc['num_layers']} != {info['num_layers']}"
            assert info_acc["target"] == info["target"], f"target: {info_acc['target']} != {info['target']}"
            assert info_acc["seed"] == info["seed"], f"seed: {info_acc['seed']} != {info['seed']}"
            try:
                info["run_id_acc"] = info_acc["run_id_acc"]
                info["run_name_acc"] = info_acc["run_name"]
                info["test_e_mae"] = info_acc["test_e_mae"]
                info["test_f_mae"] = info_acc["test_f_mae"]
                info["best_test_e_mae"] = info_acc["best_test_e_mae"]
                info["best_test_f_mae"] = info_acc["best_test_f_mae"]
            except Exception as e:
                print(f"    Error: {e}. run_name: {_run_name}")
            break
    if "test_f_mae" not in info:
        print(f"    Warning: Could not find accuracy run for {_run_name}")
    else:
        infos.append(info)

# to pandas dataframe
df = pd.DataFrame(infos)

# print('\nRuns combined:\n', df[["run_name", acc_metric, time_metric]])
print('\nRuns combined:\n', df[["run_name", "run_name_acc", acc_metric, "fpreuse_f_tol"]])


"""Rename columns"""
# rename 'model_is_deq' to 'Model'
# true -> DEQ, false -> Equiformer
df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
# rename 'model_is_deq' to 'Model'
df = df.rename(columns={"model_is_deq": "Model"})

# time_test_lowest should be lowest out of time_test and time_test_fpreuse
df["time_test_lowest"] = df.apply(lambda x: min(x["time_test"], x["time_test_fpreuse"]), axis=1)
# time_forward_per_batch_test
df["time_forward_per_batch_test_lowest"] = df.apply(lambda x: min(x["time_forward_per_batch_test"], x["time_forward_per_batch_test_fpreuse"]), axis=1)
# time_forward_total_test
df["time_forward_total_test_lowest"] = df.apply(lambda x: min(x["time_forward_total_test"], x["time_forward_total_test_fpreuse"]), axis=1)

""" Averages and filters """
print(f'\nColumns in df: {df.columns}')
do_not_average_over = ["run_name", "run_id_acc", "run_id_speed", "target", "seed", "eval_batch_size", "fpreuse_f_tol", "Model", "num_layers"]
if target == "all":
    # average over all targets
    # cols_not_avg = list(df.columns)
    # cols_not_avg.remove("target")
    # for col in do_not_average_over:
    #     if col in cols_not_avg:
    #         cols_not_avg.remove(col)
    # df = df.groupby(cols_not_avg).mean().reset_index()
    do_not_average_over.remove("target")
    df = df.groupby(do_not_average_over).mean(numeric_only=True).reset_index()
else:
    # filter out one target
    print('Filtering for target:', target)
    df = df[df["target"] == target]

# compute mean and std over 'seed'
# cols = list(df.columns)
# cols.remove("seed")
# df_mean = df.groupby(cols).mean().reset_index()
# df_std = df.groupby(cols).std().reset_index()


df = df[df["eval_batch_size"] == filter_eval_batch_size]

# fpreuse_f_tol="_default" -> 1e-3
df["fpreuse_f_tol"] = df["fpreuse_f_tol"].apply(lambda x: 1e-3 if x == "_default" else x)

# remove fpreuseftol-1e2
# df = df[df["fpreuse_f_tol"] != 1e2]
df = df[df["fpreuse_f_tol"].isin(filter_fpreuseftol)]

# for Equiformer only keep num_layers=[1,4, 8]
# df = df[df["num_layers"].isin(layers)]
df = df[
    (df["num_layers"].isin(layers_deq) & (df["Model"] == "DEQ")) | (df["num_layers"].isin(layers_equi) & (df["Model"] == "Equiformer"))
]
# isin(layers_deq) and Model=DEQ or isin(layers_equi) and Model=Equiformer
# df = df[(df["num_layers"].isin(layers_equi) & (df["Model"] == "Equiformer")) | (df["num_layers"].isin(layers_deq) & (df["Model"] == "DEQ"))]

print('\nAfter filtering:\n', df[["run_name", "run_name_acc", acc_metric, time_metric, "fpreuse_f_tol"]])


################################################################################################################################
# PLOTS
################################################################################################################################

""" Barchart of inference time """

y = time_metric
x = "run_name"
color = "Model"

# plot
# set_seaborn_style(figsize=(10, 5))
# sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df, x=x, y=y, hue=color, ax=ax)

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
name = f"inferencetime-bs{filter_eval_batch_size}-{time_metric}"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to {plotfolder}/{name}.png")

""" Barchart of accuracy """
y = acc_metric
x = "run_name"
color = "Model"

# plot
# set_seaborn_style(figsize=(10, 5))
# sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df, x=x, y=y, hue=color, ax=ax)

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
name = f"acc-bs{filter_eval_batch_size}-{acc_metric}"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to {plotfolder}/{name}.png")


""" Plot accuracy over fpreuse_f_tol """
df_fpreuse = df[df["Model"] == "DEQ"]

y = acc_metric
x = "fpreuse_f_tol"
color = "Model"

# plot
set_seaborn_style()
fig, ax = plt.subplots()
sns.scatterplot(data=df_fpreuse, x=x, y=y, hue=color, ax=ax)

# x axis on log scale
ax.set_xscale('log')
# turn around x axis
ax.invert_xaxis()
# turn on scientific notation
# ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

# labels
ax.set_xlabel("Abs solver tolerance")
ax.set_ylabel("Force MAE")
ax.set_title("Accuracy scaling with fpreuse_f_tol")

plt.tight_layout()

# save
name = f"acc_over_fpreuse_f_tol" + f"-bs{filter_eval_batch_size}-{time_metric}"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to {plotfolder}/{name}.png")


""" Plot accuracy over inference time"""

# y = "best_test_f_mae"
y = "test_f_mae"
x = time_metric
colorstyle = "Model"
shapestyle = "num_layers"
# https://stackoverflow.com/a/64403147/18361030
marks = ["o", "s", "^"]

# plot
set_seaborn_style()

fig, ax = plt.subplots()


sns.lineplot(
    data=df, x=x, y=y, hue=colorstyle, ax=ax, 
    # markers=marks[:len(list(df[shapestyle].unique()))], style=shapestyle, 
    legend=False,
)
sns.scatterplot(data=df, x=x, y=y, hue=colorstyle, style=shapestyle, ax=ax, markers=marks[:len(list(df[shapestyle].unique()))], s=200)

# # mean and std over 'seed'
# df_mean = df.groupby([x, colorstyle, shapestyle]).mean(numeric_only=True).reset_index()
# df_std = df.groupby([x, colorstyle, shapestyle]).std(numeric_only=True).reset_index()
# sns.pointplot(data=df_mean, x=x, y=y, hue=colorstyle, ax=ax, markers=marks[:len(list(df[shapestyle].unique()))], style=shapestyle, legend=False)

# sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)
# ax.errorbar(df_mean[x], df_mean[y], yerr=df_std[y], fmt='o', color='black', capsize=5)

# remove legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])

# labels
ax.set_xlabel(timelabels[x.replace("_lowest", "")])
ax.set_ylabel("Force MAE")
ax.set_title("Accuracy scaling with inference time")

# ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

plt.tight_layout(pad=0.1)

# save
name = f"acc_over_inferencetime" + f"-bs{filter_eval_batch_size}-{time_metric}"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to {plotfolder}/{name}.png")