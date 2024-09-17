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

from deq2ff.plotting.style import (
    set_seaborn_style,
    PALETTE,
    entity,
    projectmd,
    projectoc,
    plotfolder,
    set_style_after,
    myrc,
)

nans = ["NaN", pd.NA, None, float("inf"), np.nan]

def get_runs_from_wandb(
        download_data = True, # from wandb or from file
        project = projectmd,
):
    """Download runs from wandb and save to file."""
    # hosts, hostname = ["tacozoid11", "tacozoid10"], "taco"
    fname = f"runs_p_{project.replace('project', '')}"
    fullfname = f"{plotfolder}/{fname}.csv"

    if not os.path.exists(fullfname):
        download_data = True
        print(f"File {fullfname} does not exist, will download data.")
    if download_data:
        api = wandb.Api()
        runs = api.runs(
            project,
            {
                # "tags": "inference2",
                # "$and": [{"tags": "md17"}, {"tags": "eval"}],
                # "state": "finished",
                # "$or": [{"tags": "md17"}, {"tags": "main2"}, {"tags": "inference"}],
                # "state": "finished",
                "$or": [{"state": "finished"}, {"state": "crashed"}],
            },
        )
        run_ids = [run.id for run in runs]
        print(f"Found {len(run_ids)} runs:")

        infos_acc = []
        for run in runs:
            # filters
            # host = requests.get(run.file("wandb-metadata.json").url).json()["host"]
            # if host not in hosts:
            #     print(f"Skipping run {run.id} {run.name} because of host={host}")
            #     continue

            # info = {
            #     "run_id": run.id,
            #     "run_name": run.name,
            #     "eval_batch_size": run.config["eval_batch_size"],
            #     "Target": run.config["target"],
            #     "Parameters": run.summary["Model Parameters"],
            #     "seed": run.config["seed"],
            #     "num_layers": run.config["model"]["num_layers"],
            #     "model_is_deq": run.config["model_is_deq"],
            #     "evaluate": run.config["evaluate"],
            # }

            info = {
                "run_id": run.id,
                "run_name": run.name,
                # "config": run.config,
                # "summary": run.summary,
            }
            # flatten the config and summary dictionaries
            for key, value in run.config.items():
                # check if config_key is a dictionary
                if isinstance(run.config[key], dict):
                    for k2, v2 in run.config[key].items():
                        info[f"config.{key}.{k2}"] = run.config[key][k2]
                else:
                    info[f"config.{key}"] = run.config[key]
            for summary_key in run.summary.keys():
                info[f"summary.{summary_key}"] = run.summary[summary_key]

            # metadata
            # host = requests.get(run.file("wandb-metadata.json").url).json()["host"]
            # info["host"] = host

            infos_acc.append(info)

            df = pd.DataFrame(infos_acc)

        # save runs to file
        df.to_csv(fullfname, index=False)
        print("Saved dataframe.")

    else:
        # load runs from file
        df = pd.read_csv(fullfname)
        print("Loaded dataframe.")

    return df

def add_best_run(df_in, df_out, criteria, anti_criteria, metric):
    """Util function for filter_best_runs.
    Filter the best run from a dataframe based on criteria and anti-criteria.
    df_in: dataframe to filter from
    df_out: list of dataframes to append to
    """
    _df_in = df_in.copy()
    # filter by criteria
    for key, value in criteria.items():
        _df_in = _df_in[_df_in[key] == value]
    # filter by anti-criteria
    for key, value in anti_criteria.items():
        _df_in = _df_in[_df_in[key] != value]
    # select best run
    _df_in = _df_in[_df_in[metric] == _df_in[metric].min()]
    df_out.append(_df_in)
    return df_out

def filter_best_runs(
        df,
        error_metric = "summary.test_f_mae",
        criteria = {"config.model.attn_alpha_channels": 16},
        anti_criteria = {"config.dname": "ccsd"},
        deqlayers = [1, 2],
        equiformerlayers = [1, 4, 8],
    ):
    """Get the best runs from a dataframe based on criteria and anti-criteria.
    df: all runs you have
    returns: dataframe with one run per combination of target, model type, and number of layers.
    """

    df_best_runs = []

    for target in df["config.target"].unique():
        criteriatarget = criteria.copy()
        criteriatarget.update({"config.target": target})

        # best DEQ
        for deqlayer in deqlayers:
            criteria_deq = {
                "config.model_is_deq": True, "config.model.num_layers": deqlayer, 
                # "config.deq_kwargs.f_tol": 1e-2
            }
            criteria_deq.update(criteriatarget)
            df_best_runs = add_best_run(df, df_best_runs, criteria_deq, anti_criteria, error_metric)

        # best Equiformer
        for elayer in equiformerlayers:
            criteria_e = {"config.model_is_deq": False, "config.model.num_layers": elayer}
            criteria_e.update(criteriatarget)
            df_best_runs = add_best_run(df, df_best_runs, criteria_e, anti_criteria, error_metric)

    return pd.concat(df_best_runs)