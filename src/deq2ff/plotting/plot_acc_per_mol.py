import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib
import yaml

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder

# options
acc_metric = "test_f_mae"
x = "type" # "Model" run_name
runs_with_dropout = False

# get all runs with tag 'inference_speed'
api = wandb.Api()
# runs = api.runs(project, {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]})
# runs = api.runs(project, {"tags": "md17"})
runs = api.runs(project, {"$or": [{"tags": "md17"}, {"tags": "md22"}]})
run_ids = [run.id for run in runs]
print(f"Found {len(run_ids)} runs with tag 'inference_speed'")

infos = []
for run in runs:
    try:
        # model.drop_path_rate=0.05
        if runs_with_dropout:
            if run.config["model"]["drop_path_rate"] != 0.05:
                continue
        else:
            if run.config["model"]["drop_path_rate"] != 0.0:
                continue
        info = {
            "run_id": run.id,
            "run_name": run.name,
            "seed": run.config["seed"],
            "num_layers": run.config["model"]["num_layers"],
            "model_is_deq": run.config["model_is_deq"],
            "target": run.config["target"],
            "params": run.summary["Model Parameters"],
            # accuracy metrics
            "test_e_mae": run.summary["test_e_mae"],
            "test_f_mae": run.summary["test_f_mae"],
            "best_test_e_mae": run.summary["best_test_e_mae"],
            "best_test_f_mae": run.summary["best_test_f_mae"],
        }
        # Plots: pick the smaller of test_fpreuse_f_mae and test_f_mae
        if 'test_fpreuse_f_mae' in run.summary:
            info["test_f_mae"] = min(run.summary["test_f_mae"], run.summary["test_fpreuse_f_mae"])
            info["test_e_mae"] = min(run.summary["test_e_mae"], run.summary["test_fpreuse_e_mae"])
    except KeyError as e:
        print(f"Skipping run {run.id} {run.name} because of KeyError: {e}")
        continue
    infos.append(info)

# to pandas dataframe
df = pd.DataFrame(infos)

# rename 'model_is_deq' to 'Model'
# true -> DEQ, false -> Equiformer
df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
# rename 'model_is_deq' to 'Model'
df = df.rename(columns={"model_is_deq": "Model"})

print(df)

# ValueError: Unable to parse string "NaN"
# convert string 'NaN' to np.nan
df = df.replace("NaN", float("nan"))
# delete rows with NaN
df = df.dropna()
# df = df.fillna(0)

# compute mean and std over 'seed'
cols = list(df.columns)
cols.remove("seed")
# df_mean = df.groupby(cols).mean(numeric_only=True).reset_index()
# df_std = df.groupby(cols).std(numeric_only=True).reset_index()

# filter for num_layers=[1,4]
df = df[df["num_layers"].isin([1, 4])]

targets = df["target"].unique().tolist()
targets.append("all")

# new column that combines Model and num_layers
df["type"] = df["Model"] + " " + df["num_layers"].astype(str) + " layers"

accmetriclables = {
    "test_f_mae": "Force MAE (final)",
    "best_test_f_mae": "Force MAE (best)",
    "test_e_mae": "Energy MAE (final)",
    "best_test_e_mae": "Energy MAE (best)"
}

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

    """ Barchart with four quadrants """
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
    # plt.legend()

    plt.tight_layout()

    # save
    name = f"acc-{mol}-allmetrics"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to {plotfolder}/{name}.png")


    """ Simple barchart of a single metric """
    set_seaborn_style(figsize=(20, 5))
    fig, ax = plt.subplots()

    sns.barplot(data=df_mol, x=x, y=acc_metric, hue="Model", ax=ax, legend=False)
    if std is not None:
        ax.errorbar(
            x=df_mol[x], y=df_mol[acc_metric], yerr=std[acc_metric], fmt='none', ecolor='black', capsize=5
        )
    ax.set_title(f"Test f_mae for {mol}")

    # labels
    ax.set_xlabel("Molecule size")
    ax.set_ylabel(accmetriclables[acc_metric])
    ax.set_title(f"Accuracy scaling with molecule size")

    # move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # save
    name = f"acc-{mol}-{acc_metric}"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to {plotfolder}/{name}.png")



""" Plot accuracy over molecule size """

molecule_sizes = {
    'aspirin': 21,
    'benzene': 12,
    'ethanol': 9,
    'malonaldehyde': 9,
    'naphthalene': 18,
    'salicylic_acid': 16,
    'toluene': 15,
    'uracil': 12,
    'AT_AT_CG_CG': 118,
    'AT_AT': 60,
    'Ac_Ala3_NHMe': 42,
    'DHA': 56,
    'buckyball_catcher': 148,
    'dw_nanotube': 370,
    'stachyose': 87
}

df["molecule_size"] = df["target"].apply(lambda x: molecule_sizes[x])

# filter out dw_nanotube and buckyball_catcher
# dw_nanotube: only DEQ not oom
# buckyball_catcher: DEQ NaN?
df = df[~df["target"].isin(["dw_nanotube", "buckyball_catcher"])]

styletype = "target" # "num_layers" target

# for y in ["test_f_mae", "best_test_f_mae", "test_e_mae", "best_test_e_mae"]:
for y in ["test_f_mae", "test_e_mae"]:
    # plot
    set_seaborn_style(figsize=(20, 5))
    fig, ax = plt.subplots(figsize=(10,5))
    sns.scatterplot(data=df, x="molecule_size", y=y, hue="Model", style=styletype, ax=ax)

    # put in molecule names
    _texts = []
    for i, txt in enumerate(df["target"]):
        # only if not already in plot
        if txt not in _texts:
            # move text a bit to the right: +1
            ax.annotate(txt, (df["molecule_size"].iloc[i]+1, df[y].iloc[i]), fontsize=8)
            _texts.append(txt)

    # labels
    ax.set_xlabel("Molecule size")
    ax.set_ylabel(accmetriclables[y])
    ax.set_title(f"Accuracy scaling with molecule size")

    # move legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # save
    name = f"acc_over_molecule_size-{y}"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to {plotfolder}/{name}.png")
