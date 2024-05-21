import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib
import yaml
import numpy as np

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder, acclabels, timelabels

def plot_acc_all_mols(_df, _targets, _x, _y, runs_with_dropout):
    # new column that combines Model and Layers
    _df["type"] = _df["Model"] + " " + _df["Layers"].astype(str) + " layers"

    legend = True
    for mol in _targets:
        # filter by molecule
        if mol == "all":
            # average over all Target column
            df_mol = _df.groupby(["Model", "Layers", "type"]).mean(numeric_only=True).reset_index()
            std = _df.groupby(["Model", "Layers", "type"]).std(numeric_only=True).reset_index()
            df_mol["Target"] = "all"
        else:
            df_mol = _df[_df["Target"] == mol]
            std = None

        """ Barchart with four quadrants """
        # plot four bar charts side by side
        fig, ax = plt.subplots(2, 2, figsize=(10, 10))
        # fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        # left: plot f_mae
        sns.barplot(data=df_mol, x=_x, y="test_f_mae", hue="Model", ax=ax[0][0], legend=legend)
        if std is not None:
            ax[0][0].errorbar(
                x=df_mol[_x], y=df_mol["test_f_mae"], yerr=std["test_f_mae"], fmt='none', ecolor='black', capsize=5
            )
        ax[0][0].set_title(f"Test f_mae for {mol}")

        # right: plot best_f_mae
        sns.barplot(data=df_mol, x=_x, y="best_test_f_mae", hue="Model", ax=ax[0][1], legend=legend)
        if std is not None:
            ax[0][1].errorbar(
                x=df_mol[_x], y=df_mol["best_test_f_mae"], yerr=std["best_test_f_mae"], fmt='none', ecolor='black', capsize=5
            )
        ax[0][1].set_title(f"Best Test f_mae for {mol}")

        # bottom left: plot e_mae
        sns.barplot(data=df_mol, x=_x, y="test_e_mae", hue="Model", ax=ax[1][0], legend=legend)
        if std is not None:
            ax[1][0].errorbar(
                x=df_mol[_x], y=df_mol["test_e_mae"], yerr=std["test_e_mae"], fmt='none', ecolor='black', capsize=5
            )
        ax[1][0].set_title(f"Test e_mae for {mol}")

        # bottom right: plot best_e_mae
        sns.barplot(data=df_mol, x=_x, y="best_test_e_mae", hue="Model", ax=ax[1][1], legend=legend)
        if std is not None:
            ax[1][1].errorbar(
                x=df_mol[_x], y=df_mol["best_test_e_mae"], yerr=std["best_test_e_mae"], fmt='none', ecolor='black', capsize=5
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
            _ax.set_xticks(loc)
            _ax.set_xticklabels(labels, rotation=45, horizontalalignment='right', fontsize=8)

            # no legend
            # _ax.legend().set_visible(False)
            # _ax.get_legend().remove()

        # no legend
        plt.legend()

        plt.tight_layout()

        # save
        name = f"acc2-{mol}-allmetrics"
        plt.savefig(f"{plotfolder}/{name}.png")
        print(f"\nSaved plot to \n {plotfolder}/{name}.png")


        """ Simple barchart of a single metric """
        set_seaborn_style(figsize=(20, 5))
        fig, ax = plt.subplots()

        sns.barplot(data=df_mol, x=_x, y=acc_metric, hue="Model", ax=ax, legend=False)
        if std is not None:
            ax.errorbar(
                x=df_mol[_x], y=df_mol[acc_metric], yerr=std[acc_metric], fmt='none', ecolor='black', capsize=5
            )
        ax.set_title(f"Test f_mae for {mol}")

        # labels
        ax.set_xlabel("Molecule size")
        ax.set_ylabel(acclabels[acc_metric])
        ax.set_title(f"Accuracy scaling with molecule size")

        # move legend outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # save
        name = f"acc2-{mol}-{acc_metric}"
        if runs_with_dropout:
            name += "-dropout"
        else:
            name += "-nodropout"
        plt.savefig(f"{plotfolder}/{name}.png")
        print(f"\nSaved plot to \n {plotfolder}/{name}.png")



def plot_acc_over_size(_df, _y="test_f_mae", runs_with_dropout=True):
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

    _df["molecule_size"] = _df["Target"].apply(lambda x: molecule_sizes[x])

    # filter out dw_nanotube and buckyball_catcher
    # dw_nanotube: only DEQ not oom
    # buckyball_catcher: DEQ NaN?
    _df = _df[~_df["Target"].isin(["dw_nanotube", "buckyball_catcher"])]

    styletype = "Target" # "Layers" Target

    # for y in ["test_f_mae", "best_test_f_mae", "test_e_mae", "best_test_e_mae"]:
    for _y in ["test_f_mae", "test_e_mae"]:
        # plot
        set_seaborn_style(figsize=(20, 5))
        fig, ax = plt.subplots(figsize=(10,5))
        sns.scatterplot(data=_df, x="molecule_size", y=_y, hue="Model", style=styletype, ax=ax)

        # put in molecule names
        _texts = []
        for i, txt in enumerate(_df["Target"]):
            # only if not already in plot
            if txt not in _texts:
                # move text a bit to the right: +1
                ax.annotate(txt, (_df["molecule_size"].iloc[i]+1, _df[_y].iloc[i]), fontsize=8)
                _texts.append(txt)

        # labels
        ax.set_xlabel("Molecule size")
        ax.set_ylabel(acclabels[_y])
        ax.set_title(f"Accuracy scaling with molecule size")

        # move legend outside
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        # save
        name = f"acc_over_molecule_size2-{_y}"
        if runs_with_dropout:
            name += "-dropout"
        else:
            name += "-nodropout"
        plt.savefig(f"{plotfolder}/{name}.png")
        print(f"\nSaved plot to \n {plotfolder}/{name}.png")
    
def print_acc_all(_df, runs_with_dropout):
    # filter for target=aspirin
    # _df = _df[_df["Target"] == "aspirin"]
    
    # format
    # cols: Aspirin & Benzene & Ethanol & Malonaldehyde & Naphthalene & Salicylic acid & Toluene & Uracil
    # cols: energy & forces
    # rows: Equiformer 1 layer & Equiformer 4 layers & Equiformer 8 layers & DEQ 1 layer & DEQ 2 layers
    # mean \pm std
    _df["type"] = _df["Model"] + " (" + _df["Layers"].astype(str) + " layers)"
    _df["type"] = _df["type"].str.replace("1 layers", "1 layer")
    _df["type"] = _df["type"].str.replace("DEQ", "DEQuiformer")

    # cast test_f_mae and test_e_mae to float
    _df["test_f_mae"] = _df["test_f_mae"].astype(float)
    _df["test_e_mae"] = _df["test_e_mae"].astype(float)

    # for row in ["test_f_mae", "test_e_mae"]:
        # print(_df.pivot(index="type", columns="Target", values=row).to_latex(float_format="%.2f"))
    
    _df = _df.sort_values(by=["Target", "Model", "Layers"], ascending=[True, False, True])

    print('\nResult df:\n', _df[["type", "Model", "Layers", "Target", "seed", "test_f_mae", "test_e_mae"]])

    print('\nResult table:')
    first_deq = True
    for row in list(_df["type"].unique()):
        line = row + " & "
        # print(_df[_df["type"] == row].pivot(index="Target", columns="type", values="test_f_mae").to_latex(float_format="%.2f"))
        for col in list(_df["Target"].unique()):
            for subcol in ["test_f_mae", "test_e_mae"]:
                val = _df[(_df["type"] == row) & (_df["Target"] == col)][subcol].values
                # print(f'type={row}, target={col}, metric={subcol}:', val, type(val))
                # TODO: what to do with duplicates?
                if len(val) == 0:
                    mean = "NaN"
                    line += f"${mean}$ & "
                elif len(val) > 1:
                    mean = val.mean()
                    std = val.std()
                    line += f"${mean:.3f} \pm {std:.3f}$ & "
                else:
                    mean = val[0]
                    line += f"${mean:.3f}$ & "
        line = line[:-2] + "\\\\"
        line = line.replace('Equiformer', "\equiformer{}")
        if "DEQ" in row and first_deq:
            # print("\hline")
            print("\midrule[0.6pt]")
            first_deq = False
        print(line)

def print_acc(_df, runs_with_dropout, energies=False):
    # filter for target=aspirin
    # _df = _df[_df["Target"] == "aspirin"]

    # format
    # cols: Aspirin & Benzene & Ethanol & Malonaldehyde & Naphthalene & Salicylic acid & Toluene & Uracil
    # cols: energy & forces
    # rows: Equiformer 1 layer & Equiformer 4 layers & Equiformer 8 layers & DEQ 1 layer & DEQ 2 layers
    # mean \pm std
    _df["type"] = _df["Model"] + " (" + _df["Layers"].astype(str) + " layers)"
    _df["type"] = _df["type"].str.replace("1 layers", "1 layer")
    _df["type"] = _df["type"].str.replace("DEQ", "DEQuiformer")

    # cast test_f_mae and test_e_mae to float
    _df["test_f_mae"] = _df["test_f_mae"].astype(float)
    _df["test_e_mae"] = _df["test_e_mae"].astype(float)

    # for row in ["test_f_mae", "test_e_mae"]:
        # print(_df.pivot(index="type", columns="Target", values=row).to_latex(float_format="%.2f"))
    
    _df = _df.sort_values(by=["Target", "Model", "Layers"], ascending=[True, False, True])
    _df = _df.sort_values(by=["Model", "Target", "Layers"], ascending=[False, True, True])

    print('\nResult df:\n', _df[["type", "Model", "Layers", "Target", "seed", "test_f_mae", "test_e_mae"]])

    print('\nResult table:')
    first_deq = True
    lines = []
    mean_values = np.zeros((len(_df["type"].unique()), len(_df["Target"].unique())))
    if energies:
        metric = "test_e_mae"
    else:
        metric = "test_f_mae"
    for _r, row in enumerate(list(_df["type"].unique())):
        line = [row + " & "]
        # print(_df[_df["type"] == row].pivot(index="Target", columns="type", values="test_f_mae").to_latex(float_format="%.2f"))
        for _c, col in enumerate(list(_df["Target"].unique())):
            val = _df[(_df["type"] == row) & (_df["Target"] == col)][metric].values
            seeds = _df[(_df["type"] == row) & (_df["Target"] == col)]["seed"].values
            # print(f'type={row}, target={col}, metric={subcol}:', val, type(val))
            if len(val) == 0:
                mean = float("inf")
                line += [f"${mean}$ & "]
                print(f" Warning: No value for {row} and {col}")
            elif len(val) > 3:
                # TODO: what to do with duplicates?
                # if there are duplicate seeds, take the first of the duplicates
                drop_index = []
                unqiue_seeds = []
                for i, s in enumerate(seeds):
                    if s in unqiue_seeds:
                        drop_index.append(i)
                    else:
                        unqiue_seeds.append(s)
                val = np.asarray(val)
                drop_index = np.asarray(drop_index)
                val = val[~drop_index]
                mean = val.mean()
                std = val.std()
                line += [f"${mean:.3f} \pm {std:.3f}$ & "]
            elif len(val) > 1:
                mean = val.mean()
                std = val.std()
                line += [f"${mean:.3f} \pm {std:.3f}$ & "]
            else:
                mean = val[0]
                line += [f"${mean:.3f}$ & "]
            mean_values[_r, _c] = mean
        line[-1] = line[-1][:-2] + "\\\\"
        line[0] = line[0].replace('Equiformer', "\equiformer{}")
        if "DEQ" in row and first_deq == True:
            # print("\hline")
            first_deq = _r
        lines.append(line)

    # mark the best row in each column
    for _c in range(mean_values.shape[1]):
        best_row = np.argmin(mean_values[:, _c])
        # lines first column is the row name
        lines[best_row][_c+1] = "$ \\mathbf{" + lines[best_row][_c+1].replace('$', '').replace(' &', '').replace('\\\\', '') + "} $"
        if _c == mean_values.shape[1] - 1:
            lines[best_row][_c+1] += '\\\\'
        else:
            lines[best_row][_c+1] += ' &'
    
    print('\nResult table:')
    for _l, line in enumerate(lines):
        if _l == first_deq:
            print("\midrule[0.6pt]")
        print("".join(line))



if __name__ == "__main__":
    """ Options """
    acc_metric = "test_f_mae"
    x = "type" # "Model" run_name
    runs_with_dropout = False

    layers_deq = [1, 2]
    layers_equi = [1, 4, 8]

    # get all runs with tag 'inference_speed'
    api = wandb.Api()
    runs = api.runs(
        project, 
        {
            # "tags": "inference", 
            # "$or": [{"tags": "md17"}, {"tags": "md22"}, {"tags": "main2"}],
            "$or": [{"tags": "md17"}],
            # "state": "finished",
            # $or": [{"tags": "md17"}, {"tags": "main2"}, {"tags": "inference"}],
            # "state": "finished",
            # "$or": [{"state": "finished"}, {"state": "crashed"}],
        }
    )
    run_ids = [run.id for run in runs]
    print(f"Found {len(run_ids)} runs with tag 'inference_speed'")

    time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]
    acc_metrics = ["test_f_mae", "test_e_mae"]

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
            if run.summary["epoch"] < 995:
                continue
            print(' ', run.name)
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

    # to pandas dataframe
    df = pd.DataFrame(infos)

    """Rename columns"""
    # rename 'model_is_deq' to 'Model'
    # true -> DEQ, false -> Equiformer
    df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
    # rename 'model_is_deq' to 'Model'
    df = df.rename(columns={"model_is_deq": "Model"})

    # rename for prettier legends
    df = df.rename(columns={"num_layers": "Layers"})

    # time_test_lowest should be lowest out of time_test and time_test_fpreuse
    # for m in time_metrics + acc_metrics:
    #     df[f"{m}_lowest"] = df.apply(lambda x: min(x[m], x[f"{m}_fpreuse"]), axis=1)

    # only keep some layers
    df = df[
        (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ")) | (df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer"))
    ]

    # sort by Target, Model, Layers
    df = df.sort_values(by=["Target", "Model", "Layers"], ascending=[True, False, True])

    print('\nAfter filtering:\n', df[["Model", "Layers", "Target", "seed", "epoch", "PathDrop", acc_metric, "run_id"]])


    # ValueError: Unable to parse string "NaN"
    # convert string 'NaN' to np.nan
    df = df.replace("NaN", float("nan"))
    # delete rows with NaN
    df = df.dropna(subset=['test_f_mae', 'test_e_mae'])
    # df = df.fillna(0)

    # compute mean and std over 'seed'
    # cols = list(df.columns)
    # cols.remove("seed")
    # df_mean = df.groupby(cols).mean(numeric_only=True).reset_index()
    # df_std = df.groupby(cols).std(numeric_only=True).reset_index()

    targets = df["Target"].unique().tolist()
    targets.append("all")

    # plot accuracy over all molecules
    # plot_acc_all_mols(copy.deepcopy(df), targets, x, acc_metric)

    # plot accuracy over molecule size
    # plot_acc_over_size(copy.deepcopy(df))

    print_acc(copy.deepcopy(df), runs_with_dropout=runs_with_dropout)
    # print_acc(copy.deepcopy(df), energies=True, runs_with_dropout=runs_with_dropout)