import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
import copy
import os, sys, pathlib
import yaml

from deq2ff.plotting.style import set_seaborn_style, entity, project, plotfolder

# Summary metrics are the last value of your logged metrics. 
# If you’re logging metrics over time/steps then you could retrieve them using our Public API with the methods history and scan_history. 
# scan_history returns the unsampled metrics (all your steps) while history returns sampled metrics 

""" Options """
filter_eval_batch_size = 4 # 1 or 4
filter_fpreuseftol = [1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5] + ['NaN', pd.NA, None, float("inf"), np.nan]
target = "aspirin" # aspirin, all
acc_metric = "test_f_mae" # test_f_mae, test_e_mae, best_test_f_mae, best_test_e_mae
layers_deq = [1, 2]
layers_equi = [1, 4, 8]
runs_with_dropout = False

eval_batch_sizes = [1, 4]
time_metrics = ["time_test", "time_forward_per_batch_test", "time_forward_total_test"]
timelabels = {
    "time_test": "Test time for 1000 samples [s]",
    "time_forward_per_batch_test": "Forward time per batch [s]",
    "time_forward_total_test": "Total forward time for 1000 samples [s]",
}
acc_metrics = ["best_test_f_mae", "test_f_mae", "best_test_e_mae", "test_e_mae"]
acclabels = {
    "best_test_f_mae": "Best force MAE",
    "test_f_mae": r"Force MAE [meV/$\AA$]",
    "best_test_e_mae": "Best energy MAE",
    "test_e_mae": "Energy MAE",
}

""" Load data """
# get all runs with tag 'inference_speed'
api = wandb.Api()

# get runs with accuracy
runs_acc = api.runs(
    project, 
    filters={
        "$or": [{"tags": "md17"}, {"tags": "depth"}, {"tags": "inference_acc"}],
        "state": "finished"
    }
)

# get accuracy runs
print('\nAccuracy runs')
infos_acc = []
for run in runs_acc:
    # run = api.run(project + "/" + run_id)
    print(' ', run.name)
    try:
        # model.drop_path_rate=0.05
        if runs_with_dropout:
            if run.config["model"]["drop_path_rate"] != 0.05:
                continue
        else:
            if run.config["model"]["drop_path_rate"] != 0.0:
                continue
        info = {
            "run_id_acc": run.id,
            "run_name": run.name,
            "target": run.config["target"],
            "seed": run.config["seed"],
            "num_layers": run.config["model"]["num_layers"],
            "model_is_deq": run.config["model_is_deq"],
            "params": run.summary["Model Parameters"],
            # accuracy metrics
            "test_e_mae": run.summary["test_e_mae"],
            "test_f_mae": run.summary["test_f_mae"],
        }
        # Plots: pick the smaller of test_fpreuse_f_mae and test_f_mae
        if 'test_fpreuse_f_mae' in run.summary:
            info["test_f_mae"] = min(run.summary["test_f_mae"], run.summary["test_fpreuse_f_mae"])
            info["test_e_mae"] = min(run.summary["test_e_mae"], run.summary["test_fpreuse_e_mae"])
    except KeyError as e:
        print(f"Skipping run {run.id} {run.name} because of KeyError: {e}")
        continue
    if "deq_kwargs_test" in run.config:
        info["fpreuse_f_tol"] = run.config["deq_kwargs_test"]["fpreuse_f_tol"]
    # evaluate does not have best_test_e_mae and best_test_f_mae
    try:
        info["best_test_e_mae"] = run.summary["best_test_e_mae"]
        info["best_test_f_mae"] = run.summary["best_test_f_mae"]
    except KeyError as e:
        info["best_test_e_mae"] = run.summary["test_e_mae"]
        info["best_test_f_mae"] = run.summary["test_f_mae"]
    infos_acc.append(info)


df = pd.DataFrame(infos_acc)

""" Averages and filters """
# rename 'model_is_deq' to 'Model'
# true -> DEQ, false -> Equiformer
df["model_is_deq"] = df["model_is_deq"].apply(lambda x: "DEQ" if x else "Equiformer")
# rename 'model_is_deq' to 'Model'
df = df.rename(columns={"model_is_deq": "Model"})

df = df[df["target"] == target]

# df = df.groupby(do_not_average_over).mean(numeric_only=True).reset_index()

df = df[df["eval_batch_size"] == filter_eval_batch_size]

# fpreuse_f_tol="_default" -> 1e-3
df["fpreuse_f_tol"] = df["fpreuse_f_tol"].apply(lambda x: 1e-3 if x == "_default" else x)

# remove fpreuseftol-1e2
# df = df[df["fpreuse_f_tol"] != 1e2]
df = df[df["fpreuse_f_tol"].isin(filter_fpreuseftol)]

# for Equiformer only keep num_layers=[1,4, 8]
# df = df[df["num_layers"].isin(layers)]
df = df[
    (df["num_layers"].isin(layers_deq) & (df["Model"] == "DEQ")) | (df["num_layers"].isin(layers_equi) & (df["Model"] == "Equiformer"))
]

print(df)

# average over 'seed'
keep_cols = ["Model", "num_layers", "fpreuse_f_tol"]
df_mean = df.groupby(keep_cols).mean(numeric_only=True).reset_index()
df_std = df.groupby(keep_cols).std(numeric_only=True).reset_index()

################################################################################################################################
# PLOTS
################################################################################################################################

x = "num_layers"
y = acc_metric
hue = "Model"

marks = ["o", "s"]

# plot accuracy
set_seaborn_style()
fig, ax = plt.subplots()

# sns.pointplot(data=df, x=x, y=y, hue=hue, ax=ax, ci="sd", dodge=True, join=False, markers=["o", "s"], palette="colorblind")
sns.pointplot(
    data=df, x=x, y=y, hue=hue, ax=ax, markers=marks, 
    estimator="mean", 
    # errorbar method (either “ci”, “pi”, “se”, or “sd”)
    errorbar="sd", # errorbar=('ci', 95), # errorbar="sd"
    capsize=0.1,
    native_scale=True,
    linestyles=["-", "--"],
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D
    linewidth=3,
    # markeredgewidth=1, markersize=5,
)


# save
name = f"acc_over_depth-{target}"
plt.savefig(f"{plotfolder}/{name}.png")
print(f"\nSaved plot to \n {plotfolder}/{name}.png")