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

from deq2ff.plotting.style import set_seaborn_style, PALETTE, entity, project, plotfolder, acclabels, timelabels, set_style_after

nans = ['NaN', pd.NA, None, float("inf"), np.nan]


def plot_acc_over_ftol(dfc, runs_with_dropout, target):
    """ Plot accuracy over fpreuse_f_tol """
    # rename column avg_n_fsolver_steps_test_fpreuse -> nsteps
    dfc = dfc.rename(columns={"avg_n_fsolver_steps_test_fpreuse": "nsteps"})
    # dfc = dfc.rename(columns={"f_steps_to_fixed_point_test_fpreuse": "nsteps"})

    # new column nfe = nsteps * Layers
    # dfc["nfe"] = dfc["nsteps"] * dfc["Layers"]
    dfc["nfe"] = dfc["nsteps"] 

    df_fpreuse = dfc[dfc["Model"] == "DEQ"]

    x = "fpreuse_f_tol"
    color = "Layers"
    
    # y = acc_metric
    # for y in ["test_f_mae", "time_forward_per_batch_test"]:
    for y in ["test_f_mae_lowest", "time_forward_per_batch_test_lowest", "nfe"]:
        # plot
        set_seaborn_style()
        fig, ax = plt.subplots()
        # sns.scatterplot(data=df_fpreuse, x=x, y=y, hue=color, ax=ax)

        # x axis on log scale
        ax.set_xscale('log')
        # turn around x axis
        ax.invert_xaxis()

        cols_to_keep = ["Model", "Layers", "fpreuse_f_tol"]
        df_mean = df_fpreuse.groupby(cols_to_keep).mean(numeric_only=True).reset_index()
        df_std = df_fpreuse.groupby(cols_to_keep).std(numeric_only=True).reset_index()
        sns.pointplot(
            data=df_fpreuse, 
            x=x, y=y, 
            estimator="mean", 
            # errorbar="sd",
            errorbar=("ci", 95),
            hue=color, 
            ax=ax, 
            # markers=["o", "s", "^"], linestyles=["-", "--", "-."], 
            palette=PALETTE,
            markersize=3, linewidth=3,
            native_scale=True, 
            capsize=.3, # log scale
            # legend=False,
        )

        # set_style_after(ax, fs=10)

        # only major grid
        plt.grid(False)
        plt.grid(which='major', axis='y', linestyle='-', linewidth='1.0', color='lightgray')

        # ax.get_legend().get_frame().set_linewidth(0.0)
        # removes axes spines top and right
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # if there is only one layer, remove the legend
        if len(dfc["Layers"].unique()) == 1:
            ax.get_legend().remove()
        ax.get_legend().remove()

        # custom legend label
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=handles[1:], labels=["DEQ"], loc="upper right")

        # set xticks at fpreuse_f_tol 
        fpreuseftols = df_fpreuse["fpreuse_f_tol"].unique()
        ax.set_xticks(fpreuseftols)
        # TODO has to be before xscale?
        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False

        # turn on scientific notation
        # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        # from matplotlib.ticker import FuncFormatter
        # scientific_formatter = FuncFormatter(lambda x, pos: '%.1e' % x)
        # ax.yaxis.set_major_formatter(scientific_formatter)

        # labels
        ax.set_xlabel(r"Abs solver tolerance $\epsilon^{FPreuse}_{test}$")
        ax.set_title("Accuracy vs. Solver Stopping Criterion")

        plt.tight_layout()

        # save
        if "mae" in y:
            name = f"acc_over_fpreuseftol" + f"-bs{filter_eval_batch_size}"
            ax.set_ylabel(r"Force MAE [kcal/mol/$\AA$]")
        elif "nfe" in y:
            name = f"nfe_over_fpreuseftol" + f"-bs{filter_eval_batch_size}"
            ax.set_ylabel(r"Number of Solver Steps")
        else:
            name = f"time_over_fpreuseftol" + f"-bs{filter_eval_batch_size}"
            ax.set_ylabel(r"Time per Batch [s]")
        if runs_with_dropout:
            name += '-dropout'
        else:
            name += '-nodropout'
        name += "-" + target
        plt.savefig(f"{plotfolder}/{name}.png")
        print(f"\nSaved plot to \n {plotfolder}/{name}.png")


        plt.cla()
        plt.clf()   
        plt.close()

# source multiruns/tolerance_sweep_fpreuseablation.sh
def plot_acc_over_ftol_wo_fpreuse(dfc, runs_with_dropout, target):
    """ Plot accuracy over fpreuse_f_tol """
    # rename column avg_n_fsolver_steps_test_fpreuse -> nsteps
    dfc = dfc.rename(columns={"avg_n_fsolver_steps_test_fpreuse": "nsteps"})
    # dfc = dfc.rename(columns={"f_steps_to_fixed_point_test_fpreuse": "nsteps"})

    # new column nfe = nsteps * Layers
    # dfc["nfe"] = dfc["nsteps"] * dfc["Layers"]
    dfc["nfe"] = dfc["nsteps"] 

    df_fpreuse = dfc[dfc["Model"] == "DEQ"]

    x = "fpreuse_f_tol"
    color = "FPReuse"
    
    # y = acc_metric
    # for y in ["test_f_mae", "time_forward_per_batch_test"]:
    for y in ["test_f_mae_lowest", "time_forward_per_batch_test_lowest", "nfe"]:
        # plot
        set_seaborn_style()
        fig, ax = plt.subplots()
        # sns.scatterplot(data=df_fpreuse, x=x, y=y, hue=color, ax=ax)

        # x axis on log scale
        ax.set_xscale('log')
        # turn around x axis
        ax.invert_xaxis()

        cols_to_keep = ["Model", "Layers", "fpreuse_f_tol"]
        df_mean = df_fpreuse.groupby(cols_to_keep).mean(numeric_only=True).reset_index()
        df_std = df_fpreuse.groupby(cols_to_keep).std(numeric_only=True).reset_index()
        sns.pointplot(
            data=df_fpreuse, 
            x=x, y=y, 
            estimator="mean", 
            # errorbar="sd",
            errorbar=("ci", 95),
            hue=color, 
            ax=ax, 
            # markers=["o", "s", "^"], linestyles=["-", "--", "-."], 
            palette=PALETTE,
            markersize=3, linewidth=3,
            native_scale=True, 
            capsize=.3, # log scale
            # legend=False,
        )

        # set_style_after(ax, fs=10)

        # only major grid
        plt.grid(False)
        plt.grid(which='major', axis='y', linestyle='-', linewidth='1.0', color='lightgray')

        # ax.get_legend().get_frame().set_linewidth(0.0)
        # removes axes spines top and right
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # if there is only one layer, remove the legend
        if len(dfc["Layers"].unique()) == 1:
            ax.get_legend().remove()
        ax.get_legend().remove()

        # custom legend label
        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles=handles[1:], labels=["DEQ"], loc="upper right")

        # set xticks at fpreuse_f_tol 
        fpreuseftols = df_fpreuse["fpreuse_f_tol"].unique()
        ax.set_xticks(fpreuseftols)
        # TODO has to be before xscale?
        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False

        # turn on scientific notation
        # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

        # from matplotlib.ticker import FuncFormatter
        # scientific_formatter = FuncFormatter(lambda x, pos: '%.1e' % x)
        # ax.yaxis.set_major_formatter(scientific_formatter)

        # labels
        ax.set_xlabel(r"Abs solver tolerance $\epsilon^{FPreuse}_{test}$")
        ax.set_title("Accuracy vs. Solver Stopping Criterion")

        plt.tight_layout()

        # save
        if "mae" in y:
            name = f"fpreuseablation_acc_over_fpreuseftol" + f"-bs{filter_eval_batch_size}"
            ax.set_ylabel(r"Force MAE [kcal/mol/$\AA$]")
        elif "nfe" in y:
            name = f"fpreuseablation_nfe_over_fpreuseftol" + f"-bs{filter_eval_batch_size}"
            ax.set_ylabel(r"Number of Solver Steps")
        else:
            name = f"fpreuseablation_time_over_fpreuseftol" + f"-bs{filter_eval_batch_size}"
            ax.set_ylabel(r"Time per Batch [s]")
        if runs_with_dropout:
            name += '-dropout'
        else:
            name += '-nodropout'
        name += "-" + target
        plt.savefig(f"{plotfolder}/{name}.png")
        print(f"\nSaved plot to \n {plotfolder}/{name}.png")


        plt.cla()
        plt.clf()   
        plt.close()

def plot_acc_over_nfe(dfc, runs_with_dropout, target):
    # avg_n_fsolver_steps_test_fpreuse
    # f_steps_to_fixed_point_test_fpreuse

    # only plot one point
    # df = df[df["fpreuse_f_tol"].isin([1e0] + nans)]
    # select the lower
    m = "test_f_mae"
    mfp = m.replace('test', 'test_fpreuse')
    # dfc[f"{m}_lowest"] = dfc.apply(lambda x: min(x[m], x[mfp]), axis=1)

    # rename column avg_n_fsolver_steps_test_fpreuse -> nsteps
    dfc = dfc.rename(columns={"avg_n_fsolver_steps_test_fpreuse": "nsteps"})
    # dfc = dfc.rename(columns={"f_steps_to_fixed_point_test_fpreuse": "nsteps"})

    # new column nfe = nsteps * Layers
    dfc["nfe"] = dfc["nsteps"] * dfc["Layers"]

    color_palette = sns.color_palette('muted')
    color_equiformer = color_palette[0]
    color_deq = color_palette[1]
    model_to_color = {"Equiformer": color_equiformer, "DEQ": color_deq}
    equiformer_first = True

    dfc.sort_values(by=["Model", "Layers", "fpreuse_f_tol"], inplace=True, ascending=[not equiformer_first, True, True])

    df_clustered = copy.deepcopy(dfc)
    df_clustered["fpreuse_f_tol"] = df_clustered["fpreuse_f_tol"].apply(lambda x: 0.0 if np.isnan(x) else x)
    # combine cols_to_keep into one
    df_clustered["run_name"] = df_clustered.apply(lambda x: f"{x['Model']} {x['Layers']}", axis=1)
    df_clustered["run_name"] = df_clustered.apply(
        lambda x: x['run_name'] + f" {x['fpreuse_f_tol']:.0e}" if x['fpreuse_f_tol'] != 0.0 else x['run_name'], axis=1
    )
    # print('\nAfter renaming:\n', df_clustered[["run_name", "Model", "Layers", acc_metric, time_metric, "fpreuse_f_tol"]])

    cols_to_keep = ["Model", "Layers", "fpreuse_f_tol"] + ["run_name"] 
    df_mean = df_clustered.groupby(cols_to_keep).mean(numeric_only=True).reset_index()
    df_std = df_clustered.groupby(cols_to_keep).std(numeric_only=True).reset_index()
    # print('\nAfter averaging:\n', df_mean[["run_name", acc_metric, time_metric, "fpreuse_f_tol"]])

    """ Plot """
    y = acc_metric
    x = "nfe"
    colorstyle = "Model"
    shapestyle = "Layers"
    # https://stackoverflow.com/a/64403147/18361030
    # marks = ["o", "s", "^"]
    marks = ["o", "X", "^", "P"]
    # marks = ["o", "P", "^"]

    set_seaborn_style()

    fig, ax = plt.subplots()

    df_mean.sort_values(by=["Model"], inplace=True, ascending=[not equiformer_first])
    print('\nMean for acc vs NFE:\n', df_mean[[x, y, colorstyle, shapestyle]])

    # error bars on both axes
    # shades of blue
    blues = sns.color_palette("Blues", n_colors=3)
    # oranges
    oranges = sns.color_palette("Reds", n_colors=6)
    model_to_colors = {"Equiformer": blues, "DEQ": oranges}
    _i = 0
    # draws error bars and lines
    for i, m in enumerate(list(dfc["Model"].unique())):
        for _l, l in enumerate(list(dfc[shapestyle].unique())):
            _mean = copy.deepcopy(df_mean)
            _mean = _mean[_mean["Model"] == m]
            _mean = _mean[_mean[shapestyle] == l]
            _std = copy.deepcopy(df_std)
            _std = _std[_std["Model"] == m]
            _std = _std[_std[shapestyle] == l]
            ax.errorbar(
                _mean[x], 
                _mean[y], 
                xerr=_std[x], 
                yerr=_std[y], 
                # fmt='o', 
                # fmt='none', # no line
                lw=2,
                # color='black', 
                color=model_to_color[m],
                # color=model_to_colors[m][_l],
                capsize=5,
                elinewidth=2,
                capthick=2,
                # legend=False,
                alpha=0.5,
            )
            _i += 1

    # sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)
    sns.scatterplot(
        data=df_mean, x=x, y=y, hue=colorstyle, style=shapestyle, ax=ax, markers=marks[:len(list(dfc[shapestyle].unique()))], s=200, 
    )

    # labels
    ax.set_xlabel(timelabels[x.replace("_lowest", "")])
    ax.set_ylabel(r"Force MAE [kcal/mol/$\AA$]")
    ax.set_title("Inference speed vs. accuracy")

    # ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

    plt.tight_layout(pad=0.1)

    # save
    name = f"acc_over_nfe" + f"-bs{filter_eval_batch_size}-{time_metric}"
    if runs_with_dropout:
        name += '-dropout'
    else:
        name += '-nodropout'
    name += "-" + target
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot_acc_over_nfe plot to \n {plotfolder}/{name}.png")

    plt.cla()
    plt.clf()   
    plt.close()


if __name__ == "__main__":
    """ Options """
    filter_eval_batch_size = 1 # 1 or 4
    filter_fpreuseftol = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4]
    filter_fpreuseftol = {"max": 1e1, "min": 1e-4}
    filter_fpreuseftol = [1e1, 1e0, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1e-2, 1e-3, 1e-4]
    # seeds = [1]
    seeds = [1, 2, 3]
    Target = "aspirin" # aspirin, all, malonaldehyde, ethanol
    time_metric = "time_forward_per_batch_test" + "_lowest" # time_test, time_forward_per_batch_test, time_forward_total_test
    acc_metric = "test_f_mae" + "_lowest" # test_f_mae_lowest, test_f_mae, test_e_mae_lowest, test_e_mae, best_test_f_mae, best_test_e_mae
    layers_deq = [2] # [1, 2]
    layers_equi = [1, 4, 8]
    # hosts = ["tacozoid11", "tacozoid10", "andreasb-lenovo"]
    # hosts, hostname = ["tacozoid11", "tacozoid10"], "taco"
    hosts, hostname = ["andreasb-lenovo"], "bahen"
    runs_with_dropout = False

    # download data or load from file
    download_data = False

    # choose from
    eval_batch_sizes = [1, 4]
    time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]
    acc_metrics = ["test_f_mae", "test_e_mae"] # + ["best_test_f_mae", "best_test_e_mae"]
    acclabels.update({f"{k}_lowest": v for k, v in acclabels.items()})

    """ Load data """
    fname = f'acc_over_speed_2-{hostname}'
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
                    "FPReuse": run.config["fpreuse_test"],
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

    else:
        # load dataframe
        df = pd.read_csv(f"{plotfolder}/{fname}.csv")

    print('Loaded dataframe:', df.head())

    # print('\nFiltering for Target:', _Target)
    df = df[df["Target"] == Target]
    assert not df.empty, "Dataframe is empty for Target"

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

    if isinstance(filter_fpreuseftol, dict):
        df = df[(df["fpreuse_f_tol"] >= filter_fpreuseftol["min"]) & (df["fpreuse_f_tol"] <= filter_fpreuseftol["max"])]
    else:
        df = df[df["fpreuse_f_tol"].isin(filter_fpreuseftol + nans)]

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

    print('\nAfter filtering:\n', df[["Model", "Layers", "test_f_mae_lowest", "test_f_mae", "test_fpreuse_f_mae", "fpreuse_f_tol"]])


    ################################################################################################################################
    # PLOTS
    ################################################################################################################################

    # plot_acc_over_nfe(copy.deepcopy(df), runs_with_dropout=runs_with_dropout, target=Target)

    plot_acc_over_ftol(copy.deepcopy(df), runs_with_dropout=runs_with_dropout, target=Target)

    plot_acc_over_ftol_wo_fpreuse(copy.deepcopy(df), runs_with_dropout=runs_with_dropout, target=Target)
    