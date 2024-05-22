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

from deq2ff.plotting.style import set_seaborn_style, PALETTE, entity, project, plotfolder, acclabels, timelabels, set_style_after, myrc

nans = ['NaN', pd.NA, None, float("inf"), np.nan]


def print_table(_df, runs_with_dropout, mode="Force", add_nfe=False):
    assert mode in ["Force", "Energy", "Time"], f"mode={mode} not in ['Force', 'Energy', 'Time']"
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

    # reset index
    _df = _df.reset_index(drop=True)

    # print(f'\n{mode} df:\n', _df[["type", "Model", "Layers", "Target", "seed", "test_f_mae", "test_e_mae"]])

    # mean
    # dfmean = _df.groupby(["type", "Target"]).mean(numeric_only=True).reset_index()
    # print(f'\n{mode} dfmean:\n', dfmean[["type", "Target", "test_f_mae", "test_e_mae"]])

    # padding of 9 chars to compensate for 'mathbf{}'
    padding = " " * 9

    print(f'\n{mode} table (dropout={runs_with_dropout}):')
    first_deq = True
    lines = []
    nfe_lines = []
    mean_values = np.zeros((len(_df["type"].unique()), len(_df["Target"].unique())))
    if mode == "Force":
        metric = "test_f_mae"
    elif mode == "Energy":
        metric = "test_e_mae"
    else:
        metric = "time_forward_per_batch_test_lowest"
    for _r, row in enumerate(list(_df["type"].unique())):
        line = [row + " & "]
        nfe_line = []
        # print(_df[_df["type"] == row].pivot(index="Target", columns="type", values="test_f_mae").to_latex(float_format="%.2f"))
        for _c, col in enumerate(list(_df["Target"].unique())):
            val = _df[(_df["type"] == row) & (_df["Target"] == col)][metric].values
            seeds = _df[(_df["type"] == row) & (_df["Target"] == col)]["seed"].values
            nfe_val = _df[(_df["type"] == row) & (_df["Target"] == col)]["NFE"].values
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
                line += [f"${padding}{mean:.3f} \pm {std:.3f}$ & "]
                # add in the NFE to a separate list of lists
                nfe_val = np.asarray(nfe_val)
                nfe_val = nfe_val[~drop_index]
                nfe_mean = np.mean(nfe_val)
                nfe_std = np.std(nfe_val)
                if "Equiformer" in row:
                    nfe_line += [f"${padding}{int(nfe_mean)} \pm {nfe_std:.3f}$ & "]
                else:
                    nfe_line += [f"${padding}{nfe_mean:.3f} \pm {nfe_std:.3f}$ & "]
            elif len(val) > 1:
                mean = val.mean()
                std = val.std()
                line += [f"${padding}{mean:.3f} \pm {std:.3f}$ & "]
                # add in the NFE to a separate list of lists
                nfe_mean = np.mean(nfe_val)
                nfe_std = np.std(nfe_val)
                if "Equiformer" in row:
                    nfe_line += [f"${padding}{int(nfe_mean)} \pm {nfe_std:.3f}$ & "]
                else:
                    nfe_line += [f"${padding}{nfe_mean:.3f} \pm {nfe_std:.3f}$ & "]
            else:
                mean = val[0]
                line += [f"${padding}{mean:.3f}$ & "]
                # add in the NFE to a separate list of lists
                nfe_mean = nfe_val[0]
                if "Equiformer" in row:
                    nfe_line += [f"${padding}{int(nfe_mean)}$ & "]
                else:
                    nfe_line += [f"${padding}{nfe_mean:.3f}$ & "]
            
            # to calc best row
            mean_values[_r, _c] = mean
        
        line[-1] = line[-1][:-2] + "\\\\" # NFE has to come first
        line[0] = line[0].replace('Equiformer', "\equiformer{}")
        if "DEQ" in row and first_deq == True:
            # print("\hline")
            first_deq = _r
        lines.append(line)
        nfe_lines.append(nfe_line)

    # mark the best row in each column
    for _c in range(mean_values.shape[1]):
        if mode == "Time":
            # ingore the first row: Equiformer 1 layer
            # compare the second row (Equiformer 4 layers) and the fourth row (DEQ 2 layers)
            for pair in [(1, 3), (2, 4)]:
                _means = mean_values[:, _c]
                mask = np.ones(_means.shape, dtype=bool) * 1000.
                mask[pair[0]] = 0
                mask[pair[1]] = 0
                _means = _means + mask
                best_row = np.argmin(_means)
                # lines first column is the row name
                line_prev = lines[best_row][_c+1]
                line_prev = line_prev.replace('$', '').replace(' &', '').replace('\\\\', '').replace(padding, '')
                lines[best_row][_c+1] = "$ \\mathbf{" + line_prev + "} $"
                if _c == mean_values.shape[1] - 1:
                    lines[best_row][_c+1] += '\\\\'
                else:
                    lines[best_row][_c+1] += ' &'
        else:
            best_row = np.argmin(mean_values[:, _c])
            # lines first column is the row name
            line_prev = lines[best_row][_c+1]
            line_prev = line_prev.replace('$', '').replace(' &', '').replace('\\\\', '').replace(padding, '')
            lines[best_row][_c+1] = "$ \\mathbf{" + line_prev + "} $"
            if _c == mean_values.shape[1] - 1:
                lines[best_row][_c+1] += '\\\\'
            else:
                lines[best_row][_c+1] += ' &'
    
    # add in the NFE at each cell
    if mode == "Time" and add_nfe:
        for _r, line in enumerate(nfe_lines):
            for _c, cell in enumerate(line):
                lines[_r][_c+1] = cell + lines[_r][_c+1]
    
    print(f'\n{mode} table:')
    for _l, line in enumerate(lines):
        if _l == first_deq:
            print("\midrule[0.6pt]")
        print("".join(line))
    
    return

def print_table_time_forces(_df, runs_with_dropout):
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

    # reset index
    _df = _df.reset_index(drop=True)

    # print(f'\n{mode} df:\n', _df[["type", "Model", "Layers", "Target", "seed", "test_f_mae", "test_e_mae"]])

    # mean
    # dfmean = _df.groupby(["type", "Target"]).mean(numeric_only=True).reset_index()
    # print(f'\n{mode} dfmean:\n', dfmean[["type", "Target", "test_f_mae", "test_e_mae"]])

    # padding of 9 chars to compensate for 'mathbf{}'
    padding = " " * 9

    print(f'\nCombined table (dropout={runs_with_dropout}):')
    lines_both = []
    for mode in ["Force", "Time"]:
        first_deq = True
        lines = []
        mean_values = np.zeros((len(_df["type"].unique()), len(_df["Target"].unique())))
        if mode == "Force":
            metric = "test_f_mae"
        elif mode == "Energy":
            metric = "test_e_mae"
        else:
            metric = "time_forward_per_batch_test_lowest"
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
                    line += [f"${padding}{mean:.3f} \pm {std:.3f}$ & "]
                elif len(val) > 1:
                    mean = val.mean()
                    std = val.std()
                    line += [f"${padding}{mean:.3f} \pm {std:.3f}$ & "]
                else:
                    mean = val[0]
                    line += [f"${padding}{mean:.3f}$ & "]
                
                # to calc best row
                mean_values[_r, _c] = mean
            
            line[-1] = line[-1][:-2] + "\\\\" # NFE has to come first
            line[0] = line[0].replace('Equiformer', "\equiformer{}")
            if "DEQ" in row and first_deq == True:
                # print("\hline")
                first_deq = _r
            lines.append(line)

        # mark the best row in each column
        for _c in range(mean_values.shape[1]):
            if mode == "Time":
                # ingore the first row: Equiformer 1 layer
                # compare the second row (Equiformer 4 layers) and the fourth row (DEQ 2 layers)
                for pair in [(1, 3), (2, 4)]:
                    _means = mean_values[:, _c]
                    mask = np.ones(_means.shape, dtype=bool) * 1000.
                    mask[pair[0]] = 0
                    mask[pair[1]] = 0
                    _means = _means + mask
                    best_row = np.argmin(_means)
                    # lines first column is the row name
                    line_prev = lines[best_row][_c+1]
                    line_prev = line_prev.replace('$', '').replace(' &', '').replace('\\\\', '').replace(padding, '')
                    lines[best_row][_c+1] = "$ \\mathbf{" + line_prev + "} $"
                    if _c == mean_values.shape[1] - 1:
                        lines[best_row][_c+1] += '\\\\'
                    else:
                        lines[best_row][_c+1] += ' &'
            else:
                best_row = np.argmin(mean_values[:, _c])
                # lines first column is the row name
                line_prev = lines[best_row][_c+1]
                line_prev = line_prev.replace('$', '').replace(' &', '').replace('\\\\', '').replace(padding, '')
                lines[best_row][_c+1] = "$ \\mathbf{" + line_prev + "} $"
                if _c == mean_values.shape[1] - 1:
                    lines[best_row][_c+1] += '\\\\'
                else:
                    lines[best_row][_c+1] += ' &'
        
        lines_both.append(lines)
    
    lines_force = lines_both[0]
    lines_time = lines_both[1]
    lines = copy.deepcopy(lines_force)
    # combine both into one table
    for _r, line in enumerate(lines_force):
        for _c, cell in enumerate(line):
            if _c == 0: 
                # first column is the row name
                continue
            lines[_r][_c] = lines_force[_r][_c].replace('\\\\', '&') + lines_time[_r][_c]
    
    # compute relative speedupt between Equiformer 8 layers and DEQ 2 layers
    avg_speedup = 0.0
    for _c, col in enumerate(list(_df["Target"].unique())):
        val_eq8 = _df[(_df["type"] == "Equiformer (8 layers)") & (_df["Target"] == col)]["time_forward_per_batch_test_lowest"].values
        val_deq2 = _df[(_df["type"] == "DEQuiformer (2 layers)") & (_df["Target"] == col)]["time_forward_per_batch_test_lowest"].values
        if len(val_eq8) == 0 or len(val_deq2) == 0:
            print(f" Warning: No value for Equiformer 8 layers and DEQ 2 layers for {col}")
            continue
        for seed in range(1, 4):
            try:
                speedup = val_eq8[seed] / val_deq2[seed]
                print(f' Target={col} Speedup: {speedup:.2f}')
            except:
                pass
        speedup = val_eq8[0] / val_deq2[0]
        # speedup = val_deq2[0] / val_eq8[0]
        print(f'Target={col} Speedup: {speedup:.2f}')
        avg_speedup += speedup
    avg_speedup /= len(list(_df["Target"].unique()))
    print(f'Average speedup: {avg_speedup:.2f}')
    
    print(f'\nCombined table:')
    for _l, line in enumerate(lines):
        if _l == first_deq:
            print("\midrule[0.6pt]")
        print("".join(line))


if __name__ == "__main__":
    """ Options """
    filter_eval_batch_size = 1 # 1 or 4
    # filter_fpreuseftol = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    # filter_fpreuseftol = {"max": 1e1, "min": 1e-4}
    set_fpreuseftol = 2e-1
    # seeds = [1]
    seeds = [1, 2, 3]
    Target = "aspirin" # aspirin, all, malonaldehyde, ethanol
    time_metric = "time_forward_per_batch_test" + "_lowest" # time_test, time_forward_per_batch_test, time_forward_total_test
    acc_metric = "test_f_mae" + "_lowest" # test_f_mae_lowest, test_f_mae, test_e_mae_lowest, test_e_mae, best_test_f_mae, best_test_e_mae
    layers_deq = [1, 2, 3]
    layers_equi = [1, 4, 8]
    # hosts = ["tacozoid11", "tacozoid10", "andreasb-lenovo"]
    # hosts, hostname = ["tacozoid11", "tacozoid10"], "taco"
    hosts, hostname = ["andreasb-lenovo"], "bahen"

    # set_fpreuseftol = {
    #     "aspirin": 2e-1,
    #     "benzene": 2e-1,
    #     "ethanol": 1e-3,
    #     "malonaldehyde": 1e-3,
    #     "naphthalene": 1e-3,
    #     "salicylic_acid": 1e-3,
    #     "toluene": 1e-3,
    #     "uracil": 1e-3,
    # }
    # runs_with_dropout = True

    set_fpreuseftol = {
        "aspirin": 2e-1,
        "benzene": 2e-1,
        "ethanol": 1e-1,
        "malonaldehyde": 1e-1,
        "naphthalene": 1e-1,
        "salicylic_acid": 1e-1,
        "toluene": 1e-1,
        "uracil": 1e-1,
    }
    runs_with_dropout = False

    # download data or load from file
    download_data = False

    # choose from
    eval_batch_sizes = [1, 4]
    time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]
    acc_metrics = ["test_f_mae", "test_e_mae"] # + ["best_test_f_mae", "best_test_e_mae"]
    acclabels.update({f"{k}_lowest": v for k, v in acclabels.items()})

    """ Load data """
    fname = f'printplot2-{hostname}'
    if runs_with_dropout:
        fname += '-dropout'
    else:
        fname += '-nodropout'
    if download_data:
        # get all runs with tag 'inference_speed'
        api = wandb.Api()
        runs = api.runs(
            project, 
            {
                "tags": "inference2", "state": "finished",
                # $or": [{"tags": "md17"}, {"tags": "main2"}, {"tags": "inference"}],
                # "state": "finished",
                # "$or": [{"state": "finished"}, {"state": "crashed"}],
            }
        )
        run_ids = [run.id for run in runs]
        print(f"Found {len(run_ids)} runs:")

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
                    "Target": run.config["target"],
                    "Parameters": run.summary["Model Parameters"],
                    "seed": run.config["seed"],
                    "num_layers": run.config["model"]["num_layers"],
                    "model_is_deq": run.config["model_is_deq"],
                    "evaluate": run.config["evaluate"],
                }
                # if we run fpreuse only we won't have these, that's why we need to make them optional
                # summary_keys = time_metrics + acc_metrics
                # for key in summary_keys:
                #     info[key] = run.summary[key]
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
            optional_summary_keys = time_metrics + acc_metrics
            optional_summary_keys += [_m + "_fpreuse" for _m in time_metrics] + [_m + "_fpreuse" for _m in acc_metrics]
            optional_summary_keys += ["avg_n_fsolver_steps_test_fpreuse", "f_steps_to_fixed_point_test_fpreuse"]
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
        df.to_csv(f"{plotfolder}/{fname}.csv", index=False)
        print('Saved dataframe:', df)

    else:
        # load dataframe
        df = pd.read_csv(f"{plotfolder}/{fname}.csv")

    print('Loaded dataframe:', df)

    # print('\nFiltering for Target:', _Target)
    # df = df[df["Target"] == Target]
    # assert not df.empty, "Dataframe is empty for Target"

    """Rename columns"""
    # rename 'model_is_deq' to 'Model'
    # true -> DEQ, false -> Equiformer
    df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
    # rename 'model_is_deq' to 'Model'
    df = df.rename(columns={"model_is_deq": "Model"})

    # rename for prettier legends
    df = df.rename(columns={"num_layers": "Layers"})

    # cast test_fpreuse_f_mae to float
    df["test_fpreuse_f_mae"] = df["test_fpreuse_f_mae"].astype(float)

    """ If FPReuse exists, use it """
    # time_test_lowest should be lowest out of time_test and time_test_fpreuse
    # for m in time_metrics + acc_metrics:
    for m in time_metrics:
        mfp = m.replace('test', 'test_fpreuse')
        df[f"{m}_lowest"] = df.apply(lambda x: min(x[m], x[mfp]), axis=1)
    for m in acc_metrics:
        mfp = m.replace('test', 'test_fpreuse')
        df[f"{m}_lowest"] = df.apply(lambda x: x[mfp] if x[mfp] < 1000.0 else  x[m], axis=1)


    df = df[df["eval_batch_size"] == filter_eval_batch_size]
    assert not df.empty, "Dataframe is empty"

    # fpreuse_f_tol="_default" -> 1e-3
    df["fpreuse_f_tol"] = df["fpreuse_f_tol"].apply(lambda x: 1e-3 if x == "_default" else x)
    assert not df.empty, "Dataframe is empty"
    
    # only keep seeds if they are in seeds
    df = df[df["seed"].isin(seeds)]
    assert not df.empty, "Dataframe is empty"

    df["avg_n_fsolver_steps_test_fpreuse"] = df["avg_n_fsolver_steps_test_fpreuse"].apply(lambda x: 1 if x == float('inf') else x)
    df["f_steps_to_fixed_point_test_fpreuse"] = df["f_steps_to_fixed_point_test_fpreuse"].apply(lambda x: 1 if x == float('inf') else x)
    df["NFE"] = df["avg_n_fsolver_steps_test_fpreuse"] * df["Layers"]

    # fpreuse_f_tol to float
    df["fpreuse_f_tol"] = df["fpreuse_f_tol"].astype(float)

    # filter for fpreuse_f_tol per target
    if isinstance(set_fpreuseftol, dict):
        _dfts = []
        for _target, _tol in set_fpreuseftol.items():
            _dft = df[df["Target"] == _target]
            print(f' Found fpreuse_f_tol={_dft["fpreuse_f_tol"].unique()} for target={_target}')
            _dft = _dft[_dft["fpreuse_f_tol"].isin([_tol] + nans)]
            print(f'  Found {_dft.shape[0]} rows for target={_target} and fpreuse_f_tol={_tol}')
            _dfts.append(_dft)
        df = pd.concat(_dfts)
        # df = df[(df["fpreuse_f_tol"] >= filter_fpreuseftol["min"]) & (df["fpreuse_f_tol"] <= filter_fpreuseftol["max"])]
    else:
        df = df[df["fpreuse_f_tol"].isin([set_fpreuseftol] + nans)]

    # fpreuse_f_tol: replace nans with 0
    # df["fpreuse_f_tol"] = df["fpreuse_f_tol"].apply(lambda x: 0.0 if np.isnan(x) else x)

    # for Equiformer only keep Layers=[1,4, 8]
    # df = df[df["Layers"].isin(layers)]
    df = df[
        (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ")) | (df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer"))
    ]
    # isin(layers_deq) and Model=DEQ or isin(layers_equi) and Model=Equiformer
    # df = df[(df["Layers"].isin(layers_equi) & (df["Model"] == "Equiformer")) | (df["Layers"].isin(layers_deq) & (df["Model"] == "DEQ"))]
    assert not df.empty, "Dataframe is empty"

    # sort by Target, Model, Layers
    df = df.sort_values(by=["Target", "Model", "Layers"], ascending=[True, False, True])

    # print('\nAfter filtering:\n', df[["Model", "Layers", "test_f_mae_lowest", "test_f_mae", "test_fpreuse_f_mae", "fpreuse_f_tol"]])
    print('\nAfter filtering:\n', df[["Target", "Model", "Layers", "test_f_mae_lowest", "fpreuse_f_tol"]])

    ################################################################################################################################
    # PRINTS
    ################################################################################################################################

    # print_speed(df, metric="time_forward_per_batch_test_lowest")
    print_table(copy.deepcopy(df), runs_with_dropout, mode="Time", add_nfe=False)

    print_table(copy.deepcopy(df), runs_with_dropout, mode="Force", add_nfe=False)
    print_table(copy.deepcopy(df), runs_with_dropout, mode="Energy", add_nfe=False)

    print_table_time_forces(copy.deepcopy(df), runs_with_dropout)