import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder


# if __name__ == "__main__":


""" Options """
target = "aspirin" # ethanol
# layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# layers = [1, 2, 4, 8]

""" Get runs """

# run_ids = [
#     # seed=1, target=aspirin
#     "en7keqeo",
#     "89gcuv3e",
#     "jp5n1t1n",
#     "449r21m9",
#     "74diu4i3",
#     "9iit4b06",
#     "cfrmpql5",
#     "3dg5u6gb",
#     "h66aekmn",
#     # TODO: two layer DEQ
#     # TODO: average over seeds
#     # TODO: average over molecules
# ]

api = wandb.Api()
# runs = api.runs(project, {"$or": [{"config.experiment_name": "foo"}, {"config.experiment_name": "bar"}]})
# runs = api.runs(project, {"tags": "md17"})
# runs = api.runs(project, {"$or": [{"tags": "md17"}, {"tags": "md22"}]})
runs = api.runs(project, {"tags": "depth"})

infos = []

# for run_id in run_ids:
for run in runs:
    # run = api.run(project + "/" + run_id)
    try:
        info = {
            "run_id": run.id,
            "run_name": run.name,
            # "config": run.config,
            # "summary": run.summary,
            "seed": run.config["seed"],
            "num_layers": run.config["model"]["num_layers"],
            "model_is_deq": run.config["model_is_deq"],
            "target": run.config["target"],
            # "load_stats": run.config["load_stats"],
            # metrics
            "best_test_e_mae": run.summary["best_test_e_mae"],
            "best_test_f_mae": run.summary["best_test_f_mae"],
            "test_e_mae": run.summary["test_e_mae"],
            "test_f_mae": run.summary["test_f_mae"],
        }
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

# filter for target
df = df[df["target"] == target]

# filter for layers
# df = df[df["num_layers"].isin(layers)]

print('')
print(df)

# compute mean and std over 'seed'
cols = list(df.columns)
metrics_to_avg = ["best_test_e_mae", "best_test_f_mae", "test_e_mae", "test_f_mae"]
avg_over = ["seed"]
cols_to_keep = [c for c in cols if c not in avg_over + metrics_to_avg]
df_mean = df.groupby(cols_to_keep).mean(numeric_only=True).reset_index()
df_std = df.groupby(cols_to_keep).std(numeric_only=True).reset_index()

# ensure Model=Equiformer comes before Model=DEQ
df = df.sort_values("Model", ascending=False)
df_mean = df_mean.sort_values("Model", ascending=False)
df_std = df_std.sort_values("Model", ascending=False)


""" Plot """
# y = "best_test_f_mae"
y = "test_f_mae"
x = "num_layers"
color = "Model"
# https://stackoverflow.com/a/64403147/18361030
marks = ["o", "s"]

for plotstyle in ['avg', 'all']:
    # plot
    set_seaborn_style()

    fig, ax = plt.subplots()

    if plotstyle == 'all':
        sns.scatterplot(data=df, x=x, y=y, hue=color, style=color, ax=ax, markers=marks)

    else:
        sns.scatterplot(data=df_mean, x=x, y=y, hue=color, style=color, ax=ax, markers=marks)
        # sns.lineplot(data=df_mean, x=x, y=y, hue=color, ax=ax, markers=marks, legend=False)
        # ax.errorbar(df_mean[x], df_mean[y], yerr=df_std[y], fmt='o', color='black', capsize=5)

    # remove legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])

    # labels
    ax.set_xlabel("Number of Layers")
    ax.set_ylabel("Force MAE")
    ax.set_title("Accuracy scaling with depth")

    # ax.legend(labels=["DEQ", "Equiformer"], loc="upper right")

    plt.tight_layout(pad=0.1)


    # save
    name = f"acc_over_depth-{target}-{plotstyle}"
    plt.savefig(f"{plotfolder}/{name}.png")
    print(f"\nSaved plot to {plotfolder}/{name}.png")


