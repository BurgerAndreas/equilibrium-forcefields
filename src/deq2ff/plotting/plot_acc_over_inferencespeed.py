import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib
import yaml

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder


# get all runs with tag 'inference_speed'
api = wandb.Api()
runs = api.runs(project, {"tags": "inference_speed"})
run_ids = [run.id for run in runs]
print(f"Found {len(run_ids)} runs with tag 'inference_speed'")

infos = []

optional_summary_keys = ["time_test_fpreuse"]
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
        # "best_test_e_mae": run.summary["best_test_e_mae"],
        # "best_test_f_mae": run.summary["best_test_f_mae"],
        "test_e_mae": run.summary["test_e_mae"],
        "test_f_mae": run.summary["test_f_mae"],
        "time_test": run.summary["time_test"],
        "seed": run.config["seed"],
        "num_layers": run.config["model"]["num_layers"],
        "model_is_deq": run.config["model_is_deq"],
        "eval_batch_size": run.config["eval_batch_size"],
    }
    for key in optional_summary_keys:
        if key in run.summary:
            info[key] = run.summary[key]
        else:
            info[key] = float("inf")
    if "deq_kwargs_test" in run.config:
        info["fpreuse_f_tol"] = run.config["deq_kwargs_test"]["fpreuse_f_tol"]
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

print(df)

# compute mean and std over 'seed'
cols = list(df.columns)
cols.remove("seed")
# df_mean = df.groupby(cols).mean().reset_index()
# df_std = df.groupby(cols).std().reset_index()

filter_eval_batch_size = 4
df = df[df["eval_batch_size"] == filter_eval_batch_size]


""" Barchart of inference time """

y = "time_test_lowest"
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
name = "inferencetime"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to {plotfolder}/{name}.png")


""" Plot accuracy over inference time"""

# y = "best_test_f_mae"
y = "test_f_mae"
x = "time_test_lowest"
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
name = "acc_over_inferencetime"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to {plotfolder}/{name}.png")