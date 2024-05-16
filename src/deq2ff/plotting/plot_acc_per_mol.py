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
runs = api.runs(project, {"tags": "md17"})
run_ids = [run.id for run in runs]
print(f"Found {len(run_ids)} runs with tag 'inference_speed'")

infos = []
for run in runs:
    info = {
        "run_id": run.id,
        "run_name": run.name,
        "seed": run.config["seed"],
        "num_layers": run.config["model"]["num_layers"],
        "model_is_deq": run.config["model_is_deq"],
        "target": run.config["target"],
        # accuracy metrics
        "test_e_mae": run.summary["test_e_mae"],
        "test_f_mae": run.summary["test_f_mae"],
        "best_test_e_mae": run.summary["best_test_e_mae"],
        "best_test_f_mae": run.summary["best_test_f_mae"],
    }
    infos.append(info)

# to pandas dataframe
df = pd.DataFrame(infos)

# rename 'model_is_deq' to 'Model'
# true -> DEQ, false -> Equiformer
df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
# rename 'model_is_deq' to 'Model'
df = df.rename(columns={"model_is_deq": "Model"})

print(df)

# compute mean and std over 'seed'
cols = list(df.columns)
cols.remove("seed")
# df_mean = df.groupby(cols).mean().reset_index()
# df_std = df.groupby(cols).std().reset_index()

# filter for num_layers=[1,4]
df = df[df["num_layers"].isin([1, 4])]

targets = df["target"].unique().tolist()
targets.append("all")

# new column that combines Model and num_layers
df["type"] = df["Model"] + " " + df["num_layers"].astype(str) + " layers"

x = "type" # "Model" run_name

for mol in targets:
    # filter by molecule
    if mol == "all":
        # average over all target column
        df_mol = df.groupby(["Model", "num_layers", "type"]).mean(numeric_only=True).reset_index()
        std = df.groupby(["Model", "num_layers", "type"]).std(numeric_only=True).reset_index()
        df_mol["target"] = "all"
    else:
        df_mol = df[df["target"] == mol]
        std = None

    # plot four bar charts side by side
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # left: plot f_mae
    sns.barplot(data=df_mol, x=x, y="test_f_mae", hue="Model", ax=ax[0][0], legend=False)
    if std is not None:
        ax[0][0].errorbar(
            x=df_mol[x], y=df_mol["test_f_mae"], yerr=std["test_f_mae"], fmt='none', ecolor='black', capsize=5
        )
    ax[0][0].set_title(f"Test f_mae for {mol}")

    # right: plot best_f_mae
    sns.barplot(data=df_mol, x=x, y="best_test_f_mae", hue="Model", ax=ax[0][1], legend=False)
    if std is not None:
        ax[0][1].errorbar(
            x=df_mol[x], y=df_mol["best_test_f_mae"], yerr=std["best_test_f_mae"], fmt='none', ecolor='black', capsize=5
        )
    ax[0][1].set_title(f"Best Test f_mae for {mol}")

    # bottom left: plot e_mae
    sns.barplot(data=df_mol, x=x, y="test_e_mae", hue="Model", ax=ax[1][0], legend=False)
    if std is not None:
        ax[1][0].errorbar(
            x=df_mol[x], y=df_mol["test_e_mae"], yerr=std["test_e_mae"], fmt='none', ecolor='black', capsize=5
        )
    ax[1][0].set_title(f"Test e_mae for {mol}")

    # bottom right: plot best_e_mae
    sns.barplot(data=df_mol, x=x, y="best_test_e_mae", hue="Model", ax=ax[1][1], legend=False)
    if std is not None:
        ax[1][1].errorbar(
            x=df_mol[x], y=df_mol["best_test_e_mae"], yerr=std["best_test_e_mae"], fmt='none', ecolor='black', capsize=5
        )
    ax[1][1].set_title(f"Best Test e_mae for {mol}")

    for r in range(ax.shape[0]):
        for c in range(ax.shape[1]):
            _ax = ax[r][c]
            _ax.set_xlabel("")
            _ax.set_ylabel("")
            # write values on top of bars
            for p in _ax.patches:
                # skip 0.00
                if p.get_height() == 0.00:
                    continue
                _ax.annotate(
                    f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 5), 
                    textcoords='offset points', fontsize=8
                )

        # make labels vertical
        # plt.xticks(rotation=90)
        loc, labels = plt.xticks()
        _ax.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize=8)

        # no legend
        _ax.legend().set_visible(False)
        _ax.get_legend().remove()

    # no legend
    plt.legend()

    plt.tight_layout()

    # save
    name = f"acc-{mol}"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to {plotfolder}/{name}.png")