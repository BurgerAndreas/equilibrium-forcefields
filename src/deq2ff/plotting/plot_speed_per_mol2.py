import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib
import yaml
import numpy as np

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder, acclabels, timelabels



def print_speed(_df, metric="time_forward_per_batch_test_lowest"):
    # select
    _df = _df[_df["fpreuse_f_tol"].isin([1e0])]

    # format
    # cols: Aspirin & Benzene & Ethanol & Malonaldehyde & Naphthalene & Salicylic acid & Toluene & Uracil
    # cols: energy & forces
    # rows: Equiformer 1 layer & Equiformer 4 layers & Equiformer 8 layers & DEQ 1 layer & DEQ 2 layers
    # mean \pm std
    _df["type"] = _df["Model"] + " (" + _df["Layers"].astype(str) + " layers)"
    _df["type"] = _df["type"].str.replace("1 layers", "1 layer")
    _df["type"] = _df["type"].str.replace("DEQ", "DEQuiformer")

    # cast test_f_mae and test_e_mae to float
    _df[metric] = _df[metric].astype(float)

    # for row in ["test_f_mae", "test_e_mae"]:
        # print(_df.pivot(index="type", columns="Target", values=row).to_latex(float_format="%.2f"))
    
    _df = _df.sort_values(by=["Target", "Model", "Layers"], ascending=[True, False, True])

    print('\nResult df:\n', _df[["type", "Model", "Layers", "Target", "seed", "test_f_mae", "test_e_mae"]])

    print('\nResult table:')
    first_deq = True
    lines = []
    mean_values = np.zeros((len(_df["type"].unique()), len(_df["Target"].unique())))
    for _r, row in enumerate(list(_df["type"].unique())):
        line = [row + " & "]
        # print(_df[_df["type"] == row].pivot(index="Target", columns="type", values="test_f_mae").to_latex(float_format="%.2f"))
        for _c, col in enumerate(list(_df["Target"].unique())):
            val = _df[(_df["type"] == row) & (_df["Target"] == col)][metric].values
            seeds = _df[(_df["type"] == row) & (_df["Target"] == col)]["seed"].values
            # print(f'type={row}, Target={col}, metric={subcol}:', val, type(val))
            if len(val) == 0:
                mean = float("inf")
                line += [f"${mean}$ & "]
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
    
    for _l, line in enumerate(lines):
        if _l == first_deq:
            print("\midrule[0.6pt]")
        print("".join(line))


# TODO
# insert a mix of plot_acc_over_speed and plot_acc_per_mol2