import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib
import yaml

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder

# options: 
filter_eval_batch_size = 1 # 1 or 4
time_metric = "time_forward_total_test" + "_lowest" # time_test, time_forward_per_batch_test, time_forward_total_test
eval_batch_sizes = [1, 4]
time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]

# get all runs with tag 'inference_speed'
api = wandb.Api()
runs = api.runs(project, {"tags": "inference_speed"})
run_ids = [run.id for run in runs]
print(f"Found {len(run_ids)} runs with tag 'inference_speed'")

# get runs with accuracy
runs_acc = api.runs(project, {"tags": "md17"})
run_ids_acc = [run.id for run in runs_acc]
#
runs_acc = api.runs(project, {"tags": "depth"})
run_ids_acc += [run.id for run in runs_acc]
#
runs_acc = api.runs(project, {"tags": "inference_acc"})
run_ids_acc += [run.id for run in runs_acc]
print(f"Found {len(run_ids_acc)} runs with tags 'md17' or 'depth'")

infos = []

# get accuracy runs
infos_acc = []
for run_id in run_ids_acc:
    run = api.run(project + "/" + run_id)
    info = {
        "run_id": run_id,
        "run_name": run.name,
        "seed": run.config["seed"],
        "num_layers": run.config["model"]["num_layers"],
        "model_is_deq": run.config["model_is_deq"],
        # accuracy metrics
        "test_e_mae": run.summary["test_e_mae"],
        "test_f_mae": run.summary["test_f_mae"],
        "best_test_e_mae": run.summary["best_test_e_mae"],
        "best_test_f_mae": run.summary["best_test_f_mae"],
    }
    infos_acc.append(info)

optional_summary_keys = [_m + "_fpreuse" for _m in time_metrics]
for run_id in run_ids:
    # run = api.run(project + "/" + run_id)
    # api = wandb.Api()
    run = api.run(project + "/" + run_id)
    print(' ', run.name)
    # print('run_config', yaml.dump(run.config))
    # exit()
    info = {
        "run_id": run_id,
        "run_name": run.name,
        # "config": run.config,
        # "summary": run.summary,
        "seed": run.config["seed"],
        "num_layers": run.config["model"]["num_layers"],
        "model_is_deq": run.config["model_is_deq"],
        "eval_batch_size": run.config["eval_batch_size"],
        # time metrics
        "time_test": run.summary["time_test"],
        "time_forward_per_batch_test": run.summary["time_forward_per_batch_test"],
        "time_forward_total_test": run.summary["time_forward_total_test"],
    }
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
        _run_name_acc = _run_name_acc.replace(" target-ethanol", "")
        if _run_name_acc == _run_name:
            try:
                info["test_e_mae"] = info_acc["test_e_mae"]
                info["test_f_mae"] = info_acc["test_f_mae"]
                info["best_test_e_mae"] = info_acc["best_test_e_mae"]
                info["best_test_f_mae"] = info_acc["best_test_f_mae"]
            except Exception as e:
                print(f"    Error: {e}. run_name: {_run_name}")
            break
    if "test_f_mae" not in info:
        print(f"    Warning: Could not find accuracy run for {_run_name}")

    infos.append(info)

# to pandas dataframe
df = pd.DataFrame(infos)

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

print(df)

# compute mean and std over 'seed'
cols = list(df.columns)
cols.remove("seed")
# df_mean = df.groupby(cols).mean().reset_index()
# df_std = df.groupby(cols).std().reset_index()


df = df[df["eval_batch_size"] == filter_eval_batch_size]


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
    ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

# make labels vertical
# plt.xticks(rotation=90)

loc, labels = plt.xticks()
ax.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize=8)

plt.tight_layout()

# save
name = f"inferencetime-bs{filter_eval_batch_size}-{time_metric}"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to {plotfolder}/{name}.png")


""" Plot accuracy over fpreuse_f_tol """
df_fpreuse = df[df["Model"] == "DEQ"]

y = "best_test_f_mae"
x = "fpreuse_f_tol"
color = "Model"

# plot
set_seaborn_style()
fig, ax = plt.subplots()
sns.scatterplot(data=df_fpreuse, x=x, y=y, hue=color, ax=ax)

# labels
ax.set_xlabel("fpreuse_f_tol")
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
color = "Model"
# https://stackoverflow.com/a/64403147/18361030
marks = ["o", "s"]

# plot
set_seaborn_style()

fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x, y=y, hue=color, style=color, ax=ax, markers=marks)

# sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)
# ax.errorbar(df_mean[x], df_mean[y], yerr=df_std[y], fmt='o', color='black', capsize=5)

# remove legend
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])

# labels
ax.set_xlabel("Inference time (s)")
ax.set_ylabel("Force MAE")
ax.set_title("Accuracy scaling with inference time")

# ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

plt.tight_layout(pad=0.1)

# save
name = f"acc_over_inferencetime" + f"-bs{filter_eval_batch_size}-{time_metric}"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to {plotfolder}/{name}.png")