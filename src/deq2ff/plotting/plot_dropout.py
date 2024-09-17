import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib
import numpy as np

from deq2ff.plotting.style import (
    set_seaborn_style,
    PALETTE,
    entity,
    projectmd,
    plotfolder,
    timelabels,
    acclabels,
)


# if __name__ == "__main__":


""" Options """
acc_metric = "test_f_mae"
x = "Dropouts"
# averaging over all molecules won't work, since we don't have depth data for all molecules
target = "aspirin"  # ethanol aspirin
# layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# layers = [1, 2, 4, 8]
remove_single_seed_runs = True
runs_with_dropout = False
layers_deq = [1]  # [2] [1, 2]
layers_equi = [4]  # [8] [1, 4, 8]

""" Get runs """

api = wandb.Api()
# runs = api.runs("username/project", filters={"tags": {"$in": ["best"]}})
# runs = api.runs(project, {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]})
runs = api.runs(projectmd, {"tags": "drop"})
# runs = api.runs(project, {"$or": [{"tags": "md17"}, {"tags": "md22"}]})
# runs = api.runs(project, {"$and": [{"tags": "depth"}, {"state": "finished"}]})
# state finished or crashed
# runs = api.runs(project, {"$and": [{"tags": "depth"}, {"$or": [{"state": "finished"}, {"state": "crashed"}]}]})

infos = []

# for run_id in run_ids:
for run in runs:
    # run = api.run(project + "/" + run_id)
    try:
        # model.path_drop=0.05
        # if runs_with_dropout:
        #     if run.config["model"]["path_drop"] != 0.05:
        #         print(f"Skipping run {run.id} {run.name} because of path_drop={run.config['model']['path_drop']}")
        #         continue
        # else:
        #     if run.config["model"]["path_drop"] != 0.0:
        #         print(f"Skipping run {run.id} {run.name} because of path_drop={run.config['model']['path_drop']}")
        #         continue
        info = {
            "run_id": run.id,
            "run_name": run.name,
            # "config": run.config,
            # "summary": run.summary,
            "seed": run.config["seed"],
            "num_layers": run.config["model"]["num_layers"],
            "model_is_deq": run.config["model_is_deq"],
            "target": run.config["target"],
            "params": run.summary["Model Parameters"],
            "PathDropout": run.config["model"]["path_drop"],
            "AlphaDropout": run.config["model"]["alpha_drop"],
            # "load_stats": run.config["load_stats"],
            # metrics
            "best_test_e_mae": run.summary["best_test_e_mae"],
            "best_test_f_mae": run.summary["best_test_f_mae"],
            "test_e_mae": run.summary["test_e_mae"],
            "test_f_mae": run.summary["test_f_mae"],
        }
        # Plots: pick the smaller of test_fpreuse_f_mae and test_f_mae
        if "test_fpreuse_f_mae" in run.summary:
            info["test_f_mae"] = min(
                run.summary["test_f_mae"], run.summary["test_fpreuse_f_mae"]
            )
            info["test_e_mae"] = min(
                run.summary["test_e_mae"], run.summary["test_fpreuse_e_mae"]
            )
    except KeyError as e:
        print(
            f"Skipping run {run.id} {run.name} because of KeyError: {e}. (Probably run is not finished yet)"
        )
        continue
    infos.append(info)

# to pandas dataframe
df = pd.DataFrame(infos)

""" Rename """
# rename 'model_is_deq' to 'Model'
# true -> DEQ, false -> Equiformer
df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
# rename 'model_is_deq' to 'Model'
df = df.rename(columns={"model_is_deq": "Model"})

# rename for prettier legends
df = df.rename(columns={"num_layers": "Layers"})

# New column: Dropout
df["Dropouts"] = df["Model"]
# add PathDropout to Dropout
df["Dropouts"] = df["Dropouts"] + " " + df["PathDropout"].astype(str)
df["Dropouts"] = df["Dropouts"] + " " + df["AlphaDropout"].astype(str)
# replace "DEQ 0.0" with "DEQ"
df["Dropouts"] = df["Dropouts"].str.replace(" 0.05", " path")
df["Dropouts"] = df["Dropouts"].str.replace(" 0.1", " alpha")
df["Dropouts"] = df["Dropouts"].str.replace(" 0.0", "")
df["Dropouts"] = df["Dropouts"].str.replace(" path alpha", " (path, alpha)")
df["Dropouts"] = df["Dropouts"].str.replace(" path", " (path)")
df["Dropouts"] = df["Dropouts"].str.replace(" alpha", " (alpha)")
df["Dropouts"] = df["Dropouts"].str.replace(", (alpha)", ", alpha")

print(
    "\nDF after renaming:\n",
    df[["run_name", "Model", "Layers", "Dropouts", "test_f_mae", "seed"]],
)


""" Filter and statistics """

# replace np.nan with "NaN"
# df["test_f_mae"] = df["test_f_mae"].apply(lambda x: float("nan") if np.isnan(x) else x)

# drop rows where test_f_mae is NaN
# replace NaN <class 'str'> with pandas NaN
df = df.replace("NaN", float("nan"))

# df = df[[not np.isna(x) for x in df["test_f_mae"]]
df = df.dropna(subset=["test_f_mae"])

# filter for target
if target not in [None, "all"]:
    df = df[df["target"] == target]

# filter for layers
# df = df[df["Layers"].isin(layers)]
# for Equiformer only keep Layers=[1,4, 8]
df = df[
    (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ"))
    | (df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer"))
]
# isin(layers_deq) and Model=DEQ or isin(layers_equi) and Model=Equiformer
# df = df[(df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer")) | (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ"))]

# sort by Dropouts
df = df.sort_values("Dropouts", ascending=False)

print("\nBefore averaging:")
print(df[["Dropouts", "Model", "Layers", "test_f_mae", "target", "seed"]])


# compute mean and std over 'seed'
cols = list(df.columns)
# metrics_to_avg = ["best_test_e_mae", "best_test_f_mae", "test_e_mae", "test_f_mae"]
# avg_over = ["seed"]
# cols_to_keep = [c for c in cols if c not in avg_over + metrics_to_avg]
cols_to_keep = ["Dropouts", "Model", "Layers", "target"]
df_mean = df.groupby(cols_to_keep).mean(numeric_only=True).reset_index()
df_std = df.groupby(cols_to_keep).std(numeric_only=True).reset_index()

# ensure Model=Equiformer comes before Model=DEQ
df = df.sort_values("Model", ascending=False)
df_mean = df_mean.sort_values("Model", ascending=False)
df_std = df_std.sort_values("Model", ascending=False)

# remove all runs that only have one seed
if remove_single_seed_runs:
    # if they have only one seed, the std is NaN
    indices_to_remove = df_std[df_std["test_f_mae"].isna()].index
    df_mean = df_mean.drop(indices_to_remove)
    df_std = df_std.drop(indices_to_remove)

print("After averaging:")
print(df_mean)


""" Plot """
# y = "best_test_f_mae"
y = acc_metric
# x = "Dropouts"
color = "Model"
# https://stackoverflow.com/a/64403147/18361030
marks = ["o", "s"]

# plot
for orient in ["v", "h"]:
    set_seaborn_style()

    fig, ax = plt.subplots()

    # sns.scatterplot(data=df_mean, x=x, y=y, hue=color, style=color, ax=ax, markers=marks)

    # ax.errorbar(df_mean[x], df_mean[y], yerr=df_std[y], fmt='o', color='black', capsize=5)
    # sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)

    # sns.pointplot(
    #     data=df, x=x, y=y, hue=color, ax=ax, markers=marks,
    #     estimator="mean",
    #     # errorbar method (either “ci”, “pi”, “se”, or “sd”)
    #     errorbar="sd", # errorbar=('ci', 95), # errorbar="sd"
    #     capsize=0.3,
    #     native_scale=True,
    #     linestyles=["-", "--"],
    #     # https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
    #     linewidth=3,
    #     # markeredgewidth=1, markersize=5,
    # )

    # seaborn remove background grid
    # sns.despine()

    sns.barplot(
        data=df,
        x=x if orient == "v" else y,
        y=y if orient == "v" else x,
        orient=orient,
        hue=color,
        ax=ax,
        legend=False,
        width=0.5,
        gap=0.1,
    )

    # sns.catplot(
    #     data=df,
    #     x=x, y=y,
    #     # x=y, y=x, orient='h',
    #     hue=color, ax=ax,
    #     kind="bar",
    # )

    # write values on top of bars
    # for p in ax.patches:
    #     # do not write 0.00
    #     if p.get_height() == 0:
    #         continue
    #     ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=8)

    # There is now a built-in Axes.bar_label to automatically label bar containers:
    # ax = sns.barplot(x='day', y='tip', data=groupedvalues)
    # ax.bar_label(ax.containers[0]) # only 1 container needed unless using `hue`

    # For custom labels (e.g., tip bars with total_bill values), use the labels parameter:
    # ax = sns.barplot(x='day', y='tip', data=groupedvalues)
    # ax.bar_label(ax.containers[0], labels=groupedvalues['total_bill'])

    # For multi-group bar plots (i.e., with hue), there will be multiple bar containers that need to be iterated:
    # ax = sns.barplot(x='day', y='tip', hue='sex', data=df)
    # for container in ax.containers:
    #     ax.bar_label(container)

    if orient == "v":
        # make labels vertical
        # plt.xticks(rotation=90)

        loc, labels = plt.xticks()
        # ax.set_xticks(loc[::2]) # TODO: this is a hack, only show every second label
        ax.set_xticks(loc)
        ax.set_xticklabels(
            labels, rotation=45, horizontalalignment="right", fontsize=12
        )

        ax.set_xlabel("")
        ax.set_ylabel(r"Force MAE [kcal/mol/$\AA$]")

    else:
        loc, labels = plt.yticks()
        # remove yticks
        plt.yticks([], [])
        # write text at location
        for i, txt in enumerate(labels):
            ax.text(
                x=0.01, y=i - 0.4, s=txt.get_text(), ha="left", va="center", fontsize=15
            )

        # more space above the top bar
        ylim = list(ax.get_ylim())
        ylim[1] = ylim[1] * 1.5
        plt.ylim(ylim)

        ax.set_ylabel("")
        ax.set_xlabel(r"Force MAE [kcal/mol/$\AA$]")

    # remove legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[0:], labels=labels[0:])

    # labels
    ax.set_title("Accuracy vs Dropout")

    # ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

    plt.tight_layout(pad=0.1)

    # save
    name = f"dropout-{orient}-{target}"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to \n {plotfolder}/{name}.png")
