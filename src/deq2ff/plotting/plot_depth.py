import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib

# andreas-burger/EquilibriumEquiFormer
entity = "andreas-burger"
project = "EquilibriumEquiFormer"

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

def get_run_ids(entity, project, filters):
    # api = wandb.Api()
    # runs = api.runs(entity + "/" + project, filters)
    # run_ids = [run.id for run in runs]
    # return run_ids
    # filter all runs for with num_layers=3
    api = wandb.Api()
    # runs = api.runs(path=f"{ENTITY}/{PROJECT}") # can also filter here as usual
    # filters={"display_name": {"$regex": ".*label_comparison.*"}} 
    runs = api.runs(entity + "/" + project)
    # runs = api.runs(entity + "/" + project, {"Hostname": "tacozoid10"})

    # print(len(runs))
    print(runs.length)

    # run_ids = [run.id for run in runs]

    for run in runs:
        print(run.id, run.name, run.config["model.num_layers"])



def from_run_ids():

    run_ids = [
        "en7keqeo",
        "89gcuv3e",
        "jp5n1t1n",
        "449r21m9",
        "74diu4i3",
        "9iit4b06",
        "cfrmpql5",
        "3dg5u6gb",
        "h66aekmn",
    ]

    infos = []
    
    for run_id in run_ids:
        # run = api.run(project + "/" + run_id)
        api = wandb.Api()
        run = api.run(project + "/" + run_id)


        infos.append({
            "run_id": run_id,
            "run_name": run.name,
            # "config": run.config,
            # "summary": run.summary,
            "best_test_e_mae": run.summary["best_test_e_mae"],
            "best_test_f_mae": run.summary["best_test_f_mae"],
            "test_e_mae": run.summary["test_e_mae"],
            "test_f_mae": run.summary["test_f_mae"],
            "seed": run.config["seed"],
            "num_layers": run.config["model"]["num_layers"],
            "model_is_deq": run.config["model_is_deq"],
        })

    # to pandas dataframe
    df = pd.DataFrame(infos)
    print(df)

    # compute mean and std over 'seed'
    cols = list(df.columns)
    cols.remove("seed")
    # df_mean = df.groupby(cols).mean().reset_index()
    # df_std = df.groupby(cols).std().reset_index()

    # y = "best_test_f_mae"
    y = "test_f_mae"
    x = "num_layers"
    color = "model_is_deq"
    marks = ["o", "x"]

    # plot
    sns.set_style(style="whitegrid")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x, y=y, hue=color, ax=ax, markers=marks)
    
    # sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)
    # ax.errorbar(df_mean[x], df_mean[y], yerr=df_std[y], fmt='o', color='black', capsize=5)

    # labels
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Force MAE")
    ax.set_title("Number of Layers vs Test F MAE")
    # labels = ["DEQ", "FF"]
    ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")


    # save
    plt.savefig(f"{plotfolder}/depth_plot.png")
    print(f"saved plot to {plotfolder}/depth_plot.png")


if __name__ == "__main__":
    # get_run_ids(entity, project, {"config.model.num_layers": 3})
    from_run_ids()