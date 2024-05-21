import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder, set_style_after


if __name__ == "__main__":


    """ Options """
    acc_metric = "test_f_mae"
    # averaging over all molecules won't work, since we don't have depth data for all molecules
    Target = "aspirin" # ethanol aspirin
    # layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # layers = [1, 2, 4, 8]
    remove_single_seed_runs = True
    runs_with_dropout = False

    """ Get runs """

    api = wandb.Api()
    # runs = api.runs("username/project", filters={"tags": {"$in": ["best"]}})
    # runs = api.runs(project, {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]})
    # runs = api.runs(project, {"tags": "md17"})
    # runs = api.runs(project, {"$or": [{"tags": "md17"}, {"tags": "md22"}]})
    # runs = api.runs(project, {"tags": "depth"})
    # runs = api.runs(project, {"$and": [{"tags": "depth"}, {"state": "finished"}]})
    # state finished or crashed
    runs = api.runs(project, {"$and": [{"tags": "depth"}, {"$or": [{"state": "finished"}, {"state": "crashed"}]}]})

    infos = []

    # for run_id in run_ids:
    for run in runs:
        # run = api.run(project + "/" + run_id)
        try:
            # model.drop_path_rate=0.05
            if runs_with_dropout:
                if run.config["model"]["drop_path_rate"] != 0.05:
                    print(f"Skipping run {run.id} {run.name} because of drop_path_rate={run.config['model']['drop_path_rate']}")
                    continue
            else:
                if run.config["model"]["drop_path_rate"] != 0.0:
                    print(f"Skipping run {run.id} {run.name} because of drop_path_rate={run.config['model']['drop_path_rate']}")
                    continue
            info = {
                "run_id": run.id,
                "run_name": run.name,
                # "config": run.config,
                # "summary": run.summary,
                "seed": run.config["seed"],
                "Layers": run.config["model"]["num_layers"],
                "model_is_deq": run.config["model_is_deq"],
                "Target": run.config["target"],
                "Parameters": run.summary["Model Parameters"],
                # "load_stats": run.config["load_stats"],
                # metrics
                "best_test_e_mae": run.summary["best_test_e_mae"],
                "best_test_f_mae": run.summary["best_test_f_mae"],
                "test_e_mae": run.summary["test_e_mae"],
                "test_f_mae": run.summary["test_f_mae"],
            }
            # Plots: pick the smaller of test_fpreuse_f_mae and test_f_mae
            if 'test_fpreuse_f_mae' in run.summary:
                info["test_f_mae"] = min(run.summary["test_f_mae"], run.summary["test_fpreuse_f_mae"])
                info["test_e_mae"] = min(run.summary["test_e_mae"], run.summary["test_fpreuse_e_mae"])
        except KeyError as e:
            print(f"Skipping run {run.id} {run.name} because of KeyError: {e}. (Probably run is not finished yet)")
            continue
        infos.append(info)

    # to pandas dataframe
    df = pd.DataFrame(infos)

    """ Filter and statistics """
    # rename 'model_is_deq' to 'Model'
    # true -> DEQ, false -> Equiformer
    df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
    # rename 'model_is_deq' to 'Model'
    df = df.rename(columns={"model_is_deq": "Model"})

    # filter for Target
    if Target not in [None, "all"]:
        df = df[df["Target"] == Target]

    # filter for layers
    # df = df[df["Layers"].isin(layers)]

    # sort by model, layers
    df = df.sort_values(["Model", "Layers"])

    print('\nBefore averaging:')
    print(df[["run_name", "Model", "Layers", "test_f_mae", "Target"]])


    # compute mean and std over 'seed'
    cols = list(df.columns)
    # metrics_to_avg = ["best_test_e_mae", "best_test_f_mae", "test_e_mae", "test_f_mae"]
    # avg_over = ["seed"]
    # cols_to_keep = [c for c in cols if c not in avg_over + metrics_to_avg]
    cols_to_keep = ["Model", "Layers", "Target"]
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

    print('After averaging:')
    print(df_mean)


    """ Plot """
    # y = "best_test_f_mae"
    y = acc_metric
    color = "Model"
    # https://stackoverflow.com/a/64403147/18361030
    marks = ["o", "s"]

    x = "Layers"
    # plot
    set_seaborn_style()

    fig, ax = plt.subplots()

    # sns.scatterplot(
    #     data=df, x=x, y=y, hue=color, style=color, ax=ax, markers=marks
    # )

    # sns.scatterplot(data=df_mean, x=x, y=y, hue=color, style=color, ax=ax, markers=marks)

    # ax.errorbar(df_mean[x], df_mean[y], yerr=df_std[y], fmt='o', color='black', capsize=5)
    # sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)

    sns.pointplot(
        data=df, 
        x=x, y=y, hue=color, 
        ax=ax, 
        markers=marks, 
        estimator="mean", 
        # errorbar method (either “ci”, “pi”, “se”, or “sd”)
        errorbar="sd", # errorbar=('ci', 95), # errorbar="sd"
        # capsize=0.3,
        native_scale=True,
        # linestyles=["-", "--"],
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
        linewidth=3,
        # markeredgewidth=1, markersize=5,
        palette="muted",
    )

    set_style_after(ax)

    # remove legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[0:], labels=labels[0:])

    # labels
    ax.set_xlabel(f"Number of {x}")
    ax.set_ylabel(r"Force MAE [meV/$\AA$]")
    if x == "Parameters":
        ax.set_title(f"Accuracy Scaling with Parameters")
    else:
        ax.set_title(f"Accuracy Scaling with Depth")

    # ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

    plt.tight_layout(pad=0.1)

    # save
    name = f"acc_over_{x.lower()}-{Target}"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to \n {plotfolder}/{name}.png")

    # close plot
    plt.close()


    """ Plot """
    # y = "best_test_f_mae"
    y = acc_metric
    color = "Model"
    # https://stackoverflow.com/a/64403147/18361030
    marks = ["o", "s"]

    x = "Parameters"
    # plot
    set_seaborn_style()

    fig, ax = plt.subplots()

    sns.pointplot(
        data=df, 
        x=x, y=y, 
        hue=color, 
        ax=ax, 
        markers=marks, 
        estimator="mean", 
        # errorbar method (either “ci”, “pi”, “se”, or “sd”)
        errorbar="sd", # errorbar=('ci', 95), # errorbar="sd"
        # capsize=0.3,
        native_scale=True,
        # linestyles=["-", "--"],
        linestyle="",
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
        linewidth=3,
        # markeredgewidth=1, markersize=5,
        palette="muted",
    )

    set_style_after(ax)

    # sns.scatterplot(data=df_mean, x=x, y=y, hue=color, style=color, ax=ax, markers=marks)

    # ax.errorbar(df_mean[x], df_mean[y], yerr=df_std[y], fmt='o', color='black', capsize=5)
    # sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)

    # remove legend
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles=handles[0:], labels=labels[0:])

    # labels
    ax.set_xlabel(f"Number of {x}")
    ax.set_ylabel(r"Force MAE [meV/$\AA$]")
    if x == "Parameters":
        ax.set_title(f"Accuracy Scaling with Parameters")
    else:
        ax.set_title(f"Accuracy Scaling with Depth")

    # ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

    plt.tight_layout(pad=0.1)

    # save
    name = f"acc_over_{x.lower()}-{Target}"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to \n {plotfolder}/{name}.png")