import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import os, sys, pathlib

from deq2ff.plotting.style import set_seaborn_style, combine_legend, entity, project, plotfolder

# define data as a json
# https://vega.github.io/vega-lite/docs/data.html
# data = {
# "data": {
#         "values": [
#             {"x": "A", "y": 28}, {"x": "B", "y": 55}, {"x": "C", "y": 43},
#             {"x": "D", "y": 91}, {"x": "E", "y": 81}, {"x": "F", "y": 53},
#             {"x": "G", "y": 19}, {"x": "H", "y": 87}, {"x": "I", "y": 52}
#         ]
#     }
# }

# Inference
# evaluate, Bahen, torchrecordmemory
# seed=1
# bfigfw83 # E2 numlayers
# 0h76tunf
# wxk5vmp0
# c24beqi6 # DEQ
# runs are not long enough to log memory in wandb
# TODO: get memory from torchrecordmemory

# Training
# epochs=1, Bahen, torchrecordmemory
# seed=1
# ofvs1hre
# l4f8lvxc
# 5325lqya
# cwxwgxj7 # DEQ
# TODO: average over multiple runs

# Training
runs = [
# t = 20 mins
# E2 MD17 epochs-1 numlayers-8 torchrecordmemory	7214320844.8
# E2 MD17 epochs-1 numlayers-4 torchrecordmemory	5312221457.07
# DEQE2 fpcof epochs-1 torchrecordmemory	3643373431.47
# t = 10 mins
# E2 MD17 epochs-1 numlayers-8 torchrecordmemory	7214333952
# E2 MD17 epochs-1 numlayers-4 torchrecordmemory	5311740859.73
# E2 MD17 epochs-1 numlayers-1 torchrecordmemory	3923443712
# DEQE2 fpcof epochs-1 torchrecordmemory	3855024128
    {"run_id": "ofvs1hre", "gpu.process.0.memoryAllocatedBytes": 7214320844.8, "run_id_acc": "en7keqeo"},
    {"run_id": "l4f8lvxc", "gpu.process.0.memoryAllocatedBytes": 5312221457.07, "run_id_acc": "cfrmpql5"},
    {"run_id": "5325lqya", "gpu.process.0.memoryAllocatedBytes": 3923443712, "run_id_acc": "3dg5u6gb"},
    {"run_id": "cwxwgxj7", "gpu.process.0.memoryAllocatedBytes": 3855024128, "run_id_acc": "h66aekmn"} # DEQ
]

infos = []

# get 
for r in runs:
    run_id = r["run_id"]
    # run = api.run(project + "/" + run_id)
    api = wandb.Api()
    run = api.run(project + "/" + run_id)
    info = {
        "run_id": run_id,
        "run_name": run.name,
        # "config": run.config,
        # "summary": run.summary,
        "seed": run.config["seed"],
        "num_layers": run.config["model"]["num_layers"],
        "model_is_deq": run.config["model_is_deq"],
        # added by hand
        "gpu.process.0.memoryAllocatedBytes": r["gpu.process.0.memoryAllocatedBytes"]
    }
    # memory runs do not include test accuracy
    run_id = r["run_id_acc"]
    api = wandb.Api()
    run = api.run(project + "/" + run_id)
    info.update({
        "run_id_acc": run_id,
        "run_name_acc": run.name,
        "params": run.summary["Model Parameters"],
        # "config": run.config,
        # "summary": run.summary,
        "best_test_e_mae": run.summary["best_test_e_mae"],
        "best_test_f_mae": run.summary["best_test_f_mae"],
        "test_e_mae": run.summary["test_e_mae"],
        "test_f_mae": run.summary["test_f_mae"],
    })

    infos.append(info)

# to pandas dataframe
df = pd.DataFrame(infos)

# rename 'model_is_deq' to 'Model'
# true -> DEQ, false -> Equiformer
df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
# rename 'model_is_deq' to 'Model'
df = df.rename(columns={"model_is_deq": "Model"})

print(df)


""" Plot: Accuracy over GPU memory allocation """

# y = "best_test_f_mae"
y = "test_f_mae"
x = "num_layers"
x = "gpu.process.0.memoryAllocatedBytes"
colorstyle = "Model"
markerstyle = "num_layers"
# https://stackoverflow.com/a/64403147/18361030
marks = ["o", "s", "D", "8", "P", "*", "v"]

# plot
set_seaborn_style()

fig, ax = plt.subplots()
sns.scatterplot(data=df, x=x, y=y, hue=colorstyle, style=markerstyle, ax=ax, markers=marks)

# sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)
# ax.errorbar(df_mean[x], df_mean[y], yerr=df_std[y], fmt='o', color='black', capsize=5)

# remove legend title
# g.get_legend().set_title(None)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[0:], labels=labels[0:])

# labels
ax.set_xlabel("Memory allocated [bytes]")
ax.set_ylabel(r"Force MAE [meV/$\AA$]")
ax.set_title("Accuracy over GPU memory allocation")

# ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

# custom legend
from matplotlib.lines import Line2D
colorpalette = sns.color_palette()
lines = [
    Line2D([], [], marker=marks[0], color=colorpalette[1], linestyle='None'),
    Line2D([], [], marker=marks[0], color=colorpalette[0], linestyle='None'),
    Line2D([], [], marker=marks[1], color=colorpalette[0], linestyle='None'),
    Line2D([], [], marker=marks[2], color=colorpalette[0], linestyle='None'),
]
labels = ["DEQ", "Equiformer 1-layer", "Equiformer 4-layer", "Equiformer 8-layer"]

# custom legend
# handles, labels = ax.get_legend_handles_labels()
# # print("labels", labels) # ['Model', 'Equiformer', 'DEQ', 'num_layers', '1', '4', '8']
# # print('color', handles[1].get_color(), handles[2].get_color())
# lines = [
#     handles[4], # DEQ
#     handles[4], # Equiformer
#     handles[5], 
#     handles[6], 
# ]
# # https://github.com/matplotlib/matplotlib/blob/v3.8.4/lib/matplotlib/lines.py#L1053-L1063
# lines[0].set_color(handles[2].get_color()), # DEQ
# lines[1].set_color(handles[1].get_color()), # Equiformer
# lines[2].set_color(handles[1].get_color()), 
# lines[3].set_color(handles[1].get_color()), 

ax.legend(handles=lines, labels=labels, loc="upper right")


# combine legend
# colorstyle_dict = {"Equiformer": "blue", "DEQ": "orange"}
# colorstyle_dict = df[colorstyle].to_list()
# colorstyle_dict = {str(n): k for n, k in enumerate(colorstyle_dict)}
# print("colorstyle_dict", colorstyle_dict)
# combine_legend(ax, colorstyle_dict=colorstyle_dict, markerstyle=markerstyle)

plt.tight_layout(pad=0.1)

# save
name = "acc_over_malloc"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to \n {plotfolder}/{name}.png")
