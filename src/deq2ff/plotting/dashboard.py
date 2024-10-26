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
import datetime

from deq2ff.plotting.style import (
    set_seaborn_style,
    PALETTE,
    entity,
    projectmd,
    projectoc,
    plotfolder,
    set_style_after,
    myrc,
    mol_names,
    human_labels,
    marks,
    reset_plot_styles,
    cdict,
)

e1 = "E1"
e4 = "E4"
e8 = "E8"
deq1 = "DEQ1"
deq2 = "DEQ2"

nans = ["NaN", pd.NA, None, float("inf"), np.nan]


def get_runs_from_wandb(
    download_data=True,  # from wandb or from file
    project=projectmd,
    filters={
        # "tags": "inference2",
        # "$and": [{"tags": "md17"}, {"tags": "eval"}],
        # "state": "finished",
        # "$or": [{"tags": "md17"}, {"tags": "main2"}, {"tags": "inference"}],
        # "state": "finished",
        "$or": [{"state": "finished"}, {"state": "crashed"}],
    },
    fname="",
):
    """Download runs from wandb and save to file."""
    # hosts, hostname = ["tacozoid11", "tacozoid10"], "taco"
    fname = f"runs_p_{project.replace('project', '')}" + fname
    fullfname = f"{plotfolder}/{fname}.csv"

    if not os.path.exists(fullfname):
        download_data = True
        print(f"File {fullfname} does not exist, will download data.")
    if download_data:
        api = wandb.Api()
        runs = api.runs(
            project,
            filters,
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
            #     "num_layers": run.config["Class"]["num_layers"],
            #     "model_is_deq": run.config["model_is_deq"],
            #     "evaluate": run.config["evaluate"],
            # }

            info = {
                "run_id": run.id,
                "run_name": run.name,
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
        # print(infos_acc)

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
    error_metric="summary.test_f_mae",
    criteria={"config.model.attn_alpha_channels": 16},
    anti_criteria={"config.dname": "ccsd"},
    deqlayers=[1, 2],
    equiformerlayers=[1, 4, 8],
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
                "config.model_is_deq": True,
                "config.model.num_layers": deqlayer,
                # "config.deq_kwargs.f_tol": 1e-2
            }
            criteria_deq.update(criteriatarget)
            df_best_runs = add_best_run(
                df, df_best_runs, criteria_deq, anti_criteria, error_metric
            )

        # best Equiformer
        for elayer in equiformerlayers:
            criteria_e = {
                "config.model_is_deq": False,
                "config.model.num_layers": elayer,
            }
            criteria_e.update(criteriatarget)
            df_best_runs = add_best_run(
                df, df_best_runs, criteria_e, anti_criteria, error_metric
            )

    return pd.concat(df_best_runs)


def preprocess_df(df, project, error_metric):
    """Add and rename columns.
    1. remove all rows where error_metric was not logged
    2. set Class / Model column
    3. add nstep / NFE columns
    4. for OC20, set target / dname column
    5. for MD, add num_atoms of target molecule and sort
    """

    # remove all rows where nothing got logged
    print("before filter:", df.shape[0])
    if project == projectmd:
        df = df[~df["summary.epoch"].isna()]
    elif project == projectoc:
        df = df[~df["config.task.dataset"].isna()]
        # discard runs where no test error was logged
        df = df[~df[error_metric].isna()]
    else:
        raise ValueError("Unknown project")
    print("after filter:", df.shape[0])

    # Make sure you're working with a copy and not a view
    df = df.copy()

    # add a column "Model" that combines model_is_deq and num_layers
    # if model_is_deq is False, then Model is "E", else "DEQ"
    df.loc[:, "Class"] = df["config.model_is_deq"].apply(
        lambda x: "E" if not x else "DEQ"
    )
    df.loc[:, "Model"] = df["config.model_is_deq"].apply(
        lambda x: "E" if not x else "DEQ"
    ) + df["config.model.num_layers"].apply(str)
    # new column mtarget that combines target and Model
    if project == projectmd:
        df.loc[:, "mtarget"] = df["Model"] + " " + df["config.target"]
    elif project == projectoc:
        df.loc[:, "mtarget"] = df["Model"]

    # add training progress to model name
    # if project == projectmd:
    #     df["Model"] = df["Model"] + " (" + df["summary.epoch"].apply(int).apply(str) + ")"
    # elif project == projectoc:
    #     df["Model"] = df["Model"] + " (" + df["summary.train/step"].apply(
    #             lambda x: str(int(x // 100))
    #         ) + "k)"

    # n_states: replace nans with 0
    df["config.deq_kwargs.n_states"] = df["config.deq_kwargs.n_states"].replace(nans, 0)

    # add nsteps (solver steps) / NFE (number of function evaluations) as columns
    if project == projectmd:
        y = "summary.avg_n_fsolver_steps_test_fpreuse"
    elif project == projectoc:
        y = "summary.val/nstep"
    df["nstep"] = df[y]
    # For Equiformer use the number of layers as NFE
    df["NFE"] = df["nstep"]
    df["NFE"] = df["NFE"].fillna(df["config.model.num_layers"])

    # where Model is DEQ, set NFE to 2*num_layers
    df.loc[df["Model"] == "DEQ2", "NFE"] = df.loc[df["Model"] == "DEQ2", "NFE"] * 2

    df["NFE_time"] = df["NFE"] * 1.231

    # for OC20 determine how much data was used
    if project == projectoc:
        df["config.target"] = "0"  # placeholder

        # earlier runs just have data-2M in the run name
        tempdf = df[df["run_name"].str.contains("data-")]
        tempdf["config.target"] = tempdf["run_name"].apply(
            # get the first word after data- in the run name
            lambda x: x[x.find("data-") + len("data-") :].split(" ")[0]
            if "data-" in x
            else tempdf["config.target"]
        )
        # overwrite the target in the main df
        df.loc[tempdf.index, "config.target"] = tempdf["config.target"]

        # later runs have config.dataset_size, config.datasplit
        # overwrite where possible
        if "config.datasplit" in df.columns:
            # fill in config.target where config.datasplit is not nan else leave it
            df["config.target"] = df["config.datasplit"].fillna(df["config.target"])

        # if "config.dataset_size" in df.columns:
        #     # is independent of maxdata
        #     df["config.target"] = df["config.dataset_size"].apply(
        #         lambda x: f"{int(x / 1000)}k" if pd.notna(x) else df["config.target"]
        #     )

        # where target=0, fill in "200k" as default
        print(
            f"replacing {df[df['config.target'] == '0'].shape[0]}/{df.shape[0]} 0's with 200k"
        )
        df["config.target"] = df["config.target"].replace("0", "200k")
        # df["config.target"] = df["config.target"].apply(lambda x: f"200k" if x == '0' else x)

        # config.optim.maxdata
        # print(f"Adding maxdata to target")
        # if "config.optim.maxdata" in df.columns:
        #     # where maxdata is > 0, append it to the target
        #     tempdf = df[df["config.optim.maxdata"] > 0]
        #     tempdf["config.target"] = tempdf["config.optim.maxdata"].apply(
        #         lambda x: f"{int(x / 1000)}k/"  # if x > 0 else ""
        #     )
        #     df.loc[tempdf.index, "config.target"] = (
        #         tempdf["config.target"] + df["config.target"]
        #     )

        print(f"Adding optim.max_epochs to target")
        df["config.target"] = (
            df["config.target"]
            + " e"
            + df["config.optim.max_epochs"].apply(int).apply(str)
        )

    """
    SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    Chained assignment: The use of more than one indexing operation back-to-back & a setting operation.
    """

    if project == projectoc:
        df["config.dname"] = "oc20"

    # add num_atoms of target molecule and sort
    if project == projectmd:
        # load datasets/statistics.json
        # equilibrium-forcefields/src/deq2ff/plotting/dashboard.ipynb
        # equilibrium-forcefields/datasets/statistics.json
        # get folder of ipynb
        fpath = "/ssd/gen/equilibrium-forcefields/datasets/statistics.json"
        # print(fpath)
        with open(fpath, "r") as f:
            statistics = json.load(f)

        # print(statistics)
        statistics = {**statistics["md17"], **statistics["md22"]}
        # print(statistics)

        # add num_atoms to df based on target
        max_radius = df["config.model.max_radius"].unique()[0]
        max_radius = str(float(max_radius))

        # loop over rows
        # for i, row in df.iterrows():
        #     target = row["config.target"]
        #     print(i, target)
        #     num_atoms = statistics[target]
        #     num_atoms = num_atoms[max_radius]
        #     num_atoms = num_atoms["avg_node"]
        #     df.at[i, "num_atoms"] = num_atoms
        df["num_atoms"] = df["config.target"].apply(
            lambda x: statistics[x][max_radius]["avg_node"]
        )

        # sort target by num_atoms
        df = df.sort_values(["num_atoms", "config.target"])

    return df


def mark_sota(
    _df,
    comparison="pairwise",
    error_metric="summary.test_f_mae",
    sotaname="sota",
):
    """Add a column 'sota' to the dataframe that marks the best run for each target."""

    if comparison in [False, None]:
        pass

    elif comparison in "best":
        # mark the run with the lowest error_metric for each target
        _df[sotaname] = False
        for target in _df["config.target"].unique():
            _df_best = _df[_df["config.target"] == target]
            _df_best = _df_best[_df_best[error_metric] == _df_best[error_metric].min()]
            _df.loc[_df_best.index, sotaname] = True

    elif comparison in "pairwise":
        print("Models:", _df["Model"].unique())

        # we compare models of similar inference time
        # first pick out E4 and DEQ1 models and mark the best one
        _df[sotaname] = False
        for target in _df["config.target"].unique():
            _df_best = _df[_df["config.target"] == target]
            _df_best = _df_best[_df_best["Model"].isin([e4, deq1])]
            _df_best = _df_best[_df_best[error_metric] == _df_best[error_metric].min()]
            _df.loc[_df_best.index, sotaname] = True

        # then pick out E8 and DEQ2 models and mark the best one
        for target in _df["config.target"].unique():
            _df_best = _df[_df["config.target"] == target]
            _df_best = _df_best[_df_best["Model"].isin([e8, deq2])]
            _df_best = _df_best[_df_best[error_metric] == _df_best[error_metric].min()]
            _df.loc[_df_best.index, sotaname] = True

    else:
        raise ValueError(f"Unknown comparison {comparison}")

    return _df

def merge_dfs(dfog, dfextra, oncol, cols, coltypes={}):
    """Merge two dataframes. 
    Match on column oncol and add columns cols from dfextra to dfog.
    Assumes datatypes are float, unless specified in coltypes.
    """

    dfog = dfog.reset_index(drop=True)
    dfextra = dfextra.reset_index(drop=True)

    print(f"Before mergin: {dfog.shape}")
    print(f" mergin: {dfextra.shape}")
    
    # TODO: better way to merge?
    for _c in cols:
        # set dtype to float
        # not setting the same dtype can lead to unexpected nans
        if _c in coltypes:
            _type = coltypes[_c]
        else:
            _type = float
        dfog[_c] = np.nan
        dfextra[_c] = dfextra[_c].astype(_type)
        dfog[_c] = dfog[_c].astype(_type)

        # merge by hand
        # df_best_runs[_t] = np.nan
        # df_best_runs["sotatime"] = np.nan
        # for i, row in df_best_runs.iterrows():
        #     # dfspeed[dfspeed["mtarget"] == "DEQ1 ethanol"]["summary.time_forward_per_batch_test_fpreuse"]

        #     df_best_runs[_t] = dfspeed[_t].where(dfspeed[ind] == row[ind], df_best_runs[_t])
        #     df_best_runs["sotatime"] = dfspeed["sotatime"].where(dfspeed[ind] == row[ind], df_best_runs["sotatime"])
        #     # print("row", row[ind],_t, dfspeed[_t].where(dfspeed[ind] == row[ind], df_best_runs[_t]))
        #     # print("sotatime", dfspeed["sotatime"].where(dfspeed[ind] == row[ind], df_best_runs["sotatime"]))

        # for i, row in df_best_runs.iterrows():
        #     # print df_best_runs ind
        #     print("row", row[ind], df_best_runs[df_best_runs[ind] == row[ind]][_t].item(), df_best_runs[df_best_runs[ind] == row[ind]][_t].index, "<-", dfspeed[dfspeed[ind] == row[ind]][_t].item())

        #     df_best_runs[df_best_runs[ind] == row[ind]][_t] = dfspeed[dfspeed[ind] == row[ind]][_t]
        #     df_best_runs[df_best_runs[ind] == row[ind]]["sotatime"] = dfspeed[dfspeed[ind] == row[ind]]["sotatime"]

        # print(df_best_runs[ind].unique(), len(df_best_runs[ind].unique()))
        # print(dfspeed[ind].unique(), len(dfspeed[ind].unique()))

        # merge: add time to best_runs
        # df_best_runs = df_best_runs.merge(
        #     dfspeed[[ind, _t, "sotatime"]],
        #     on=ind,
        #     how="left",
        # )
        # df_best_runs = df_best_runs.reset_index(drop=True)

        # df_best_runs[_t] = df_best_runs[_t].map(dfspeed.set_index(ind)[_t])

        # Loop over each row in df2
        for index2, row2 in dfog.iterrows():
            found = False
            # Loop over each row in df1
            for index1, row1 in dfextra.iterrows():
                # If column B matches, copy value from column A in df1 to df2
                if row2[oncol] == row1[oncol]:
                    # dfog.at[index2, "sotatime"] = row1["sotatime"]
                    dfog.at[index2, _c] = row1[_c]
                    for _c in cols:
                        dfog.at[index2, _c] = row1[_c]
                    found = True
                    break
            if not found:
                dfog.at[index2, _c] = np.nan
                print(f"Could not find {row2[oncol]} in dfspeed")

    print(f"After merging: {dfog.shape}")
    return dfog

def print_table_acc( 
    df,
    error_metric,
    corner=True,
    mname="Force",
    dnames=["oc20"],
    ex_targets=["dw_nanotube"],
    models=["E1", "E4", "E8", "DEQ1", "DEQ2"],
):
    _df = df.copy()

    # select dname=md17
    _df = _df[_df["config.dname"].isin(dnames)]

    if corner == True:
        corner = " ".join([_n.upper() for _n in dnames])
    elif not isinstance(corner, str):
        corner = ""

    # exclude dw_nanotube
    for ex_target in ex_targets:
        _df = _df[_df["config.target"] != ex_target]
    # _df = _df[_df["config.target"] != "dw_nanotube"]

    targets = list(_df["config.target"].unique())
    num_targets = len(targets)

    lines = []

    # first lines: header
    r"""
    \begin{tabular}{lcccccccccccccccc}
    \toprule[1.2pt]
     & \multicolumn{2}{c}{Aspirin} & \multicolumn{2}{c}{Benzene} & \multicolumn{2}{c}{Ethanol} & \multicolumn{2}{c}{Malonaldehyde} & \multicolumn{2}{c}{Naphthalene} & \multicolumn{2}{c}{Salicylic acid} & \multicolumn{2}{c}{Toluene} & \multicolumn{2}{c}{Uracil} \\
    \cmidrule[0.6pt]{2-17}
    Methods & Forces & Time & Forces & Time & Forces & Time & Forces& Time & Forces   & Time   & Forces   & Time   & Forces  & Time  & Forces  & Time  \\
    \midrule[1.2pt]
    """
    lines += [r"\begin{tabular}{l" + (r"c" * num_targets) + r"}"]
    lines += [r"\toprule[1.2pt]"]
    # if there is more than one target, add a line with the target names
    if num_targets > 1:
        lines += [
            r" & "
            + r" & ".join([r"\multicolumn{1}{c}{" + f"{mol_names[t]}"  + r"}" for t in targets])
            + r" \\"
        ]
        lines += [r"\cmidrule[0.6pt]{2-" + f"{int(num_targets + 1)}" + r"}"]
    lines += [corner + r" & " + r" & ".join([mname for t in targets]) + r" \\"]
    lines += [r"\midrule[1.2pt]"]

    # lines with results
    for _r, row in enumerate(models):
        # mark DEQ with a horizontal line
        if row == "DEQ1":
            lines += [r"\midrule[0.6pt]"]
        # rename Model
        rname = row.replace("E", r"\equiformer{} ")
        rname = rname.replace(r"D\equiformer{} Q", r"DEQ")
        rname = rname.replace(r"DEQ", r"DEQ ")
        rname = rname.split(" ")
        rname = rname[0] + r" (" + rname[1] + r" layers)"
        rname = rname.replace(r"1 layers", r"1 layer")
        line = f"{rname}"
        # add results
        for t in targets:
            _df_t = _df[_df["config.target"] == t]
            _df_t = _df_t[_df_t["Model"] == row]
            if len(_df_t) == 0:
                line += r" & "
            else:
                # accuracy
                _err = f"{_df_t[error_metric].iloc[0]:.3f}"
                if _df_t["sota"].values[0]:
                    line += r" & $\mathbf{" + _err + r"}$"
                else:
                    line += r" & $" + _err + r"$"
        # newline
        line += r" \\"
        lines += [line]

    # print(' ')
    print("\n".join(lines))
    # print(' ')
    return lines

def print_table_acc_time(
    df,
    error_metric,
    time_metric,
    dnames=["md17"],
    corner=True,
    ex_targets=["dw_nanotube"],
    models=["E1", "E4", "E8", "DEQ1", "DEQ2"],
):
    _df = df.copy()

    # select dname=md17
    _df = _df[_df["config.dname"].isin(dnames)]
    
    if corner == True:
        corner = " ".join([_n.upper() for _n in dnames])
    elif not isinstance(corner, str):
        corner = ""

    # exclude dw_nanotube
    for ex_target in ex_targets:
        _df = _df[_df["config.target"] != ex_target]
    # _df = _df[_df["config.target"] != "dw_nanotube"]

    targets = list(_df["config.target"].unique())
    num_targets = len(targets)

    lines = []

    # first lines: header
    r"""
    \begin{tabular}{lcccccccccccccccc}
    \toprule[1.2pt]
     & \multicolumn{2}{c}{Aspirin} & \multicolumn{2}{c}{Benzene} & \multicolumn{2}{c}{Ethanol} & \multicolumn{2}{c}{Malonaldehyde} & \multicolumn{2}{c}{Naphthalene} & \multicolumn{2}{c}{Salicylic acid} & \multicolumn{2}{c}{Toluene} & \multicolumn{2}{c}{Uracil} \\
    \cmidrule[0.6pt]{2-17}
    Methods & Forces & Time & Forces & Time & Forces & Time & Forces& Time & Forces   & Time   & Forces   & Time   & Forces  & Time  & Forces  & Time  \\
    \midrule[1.2pt]
    """
    lines += [r"\begin{tabular}{l" + (r"c" * (num_targets * 2)) + r"}"]
    lines += [r"\toprule[1.2pt]"]
    # if there is more than one target, add a line with the target names
    if num_targets > 1:
        lines += [
            r" & "
            + r" & ".join([r"\multicolumn{2}{c}{" + f"{mol_names[t]}"  + r"}" for t in targets])
            + r" \\"
        ]
        lines += [r"\cmidrule[0.6pt]{2-" + f"{int(num_targets*2 + 1)}" + r"}"]
    lines += [corner + r" & " + r" & ".join([r"Force & Time" for t in targets]) + r" \\"]
    lines += [r"\midrule[1.2pt]"]

    # lines with results
    for _r, row in enumerate(models):
        # mark DEQ with a horizontal line
        if row == "DEQ1":
            lines += [r"\midrule[0.6pt]"]
        # rename Model
        rname = row.replace("E", r"\equiformer{} ")
        rname = rname.replace(r"D\equiformer{} Q", r"DEQ")
        rname = rname.replace(r"DEQ", r"DEQ ")
        rname = rname.split(" ")
        rname = rname[0] + r" (" + rname[1] + r" layers)"
        rname = rname.replace(r"1 layers", r"1 layer")
        line = f"{rname}"
        # add results
        for t in targets:
            _df_t = _df[_df["config.target"] == t]
            _df_t = _df_t[_df_t["Model"] == row]
            if len(_df_t) == 0:
                line += r" & & "
            else:
                # accuracy
                _err = f"{_df_t[error_metric].iloc[0]:.3f}"
                if _df_t["sota"].values[0]:
                    line += r" & $\mathbf{" + _err + r"}$"
                else:
                    line += r" & $" + _err + r"$"

                # timing
                _time = f"{float(_df_t[time_metric].iloc[0]):.3f}"
                if _df_t["sotatime"].values[0]:
                    line += r" & $\mathbf{" + _time + r"}$"
                else:
                    line += r" & $" + _time + r"$"
        # newline
        line += r" \\"
        lines += [line]

    # print(' ')
    print("\n".join(lines))
    # print(' ')
    return lines

def norm_targets(data, norm_scheme="minmax", error_metric="summary.test_f_mae"):
    """Normalize all target molecules to be in the same range.
    Modifies in place.
    """
    # data = normalize(data, norm_scheme=norm_scheme)
    # normalize
    ylabel = error_metric
    if norm_scheme == "minmax":
        data[error_metric] = data.groupby("config.target")[error_metric].transform(
            lambda x: (x - x.min()) / (x.max() - x.min())
        )
        ylabel = human_labels(error_metric) + " (minmax)"
    elif norm_scheme == "minmax2":
        # instead of [0, 1], normalize to [-1, 1]
        data[error_metric] = data.groupby("config.target")[error_metric].transform(
            lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1
        )
        ylabel = human_labels(error_metric) + " (normed to [-1, 1])"

    elif norm_scheme == "minmax3":
        # normalize to [0.1, 1]
        data[error_metric] = data.groupby("config.target")[error_metric].transform(
            lambda x: 0.9 * (x - x.min()) / (x.max() - x.min()) + 0.1
        )
        ylabel = human_labels(error_metric) + " (normed to [0.1, 1])"

    elif norm_scheme == "mean":
        data[error_metric] = data.groupby("config.target")[error_metric].transform(
            lambda x: x / x.mean()
        )
        ylabel = human_labels(error_metric) + " (zero mean)"
    elif norm_scheme == "zscore":
        data[error_metric] = data.groupby("config.target")[error_metric].transform(
            lambda x: (x - x.mean()) / x.std()
        )
        ylabel = human_labels(error_metric) + " (zscore)"
    elif norm_scheme == "zscoree4":
        # instead of substracing the mean, substract E4
        for t in data["config.target"].unique():
            _mean = data[(data["config.target"] == t) & (data["Model"] == "E4")][
                error_metric
            ].mean()
            _std = data[data["config.target"] == t][error_metric].std()
            data.loc[data["config.target"] == t, error_metric] = (
                data[data["config.target"] == t][error_metric] - _mean
            ) / _std
        ylabel = human_labels(error_metric) + " (zscore around mean=E4)"

    elif norm_scheme == "zscoree4":
        # instead of substracing the mean, substract E4
        for t in data["config.target"].unique():
            _mean = data[(data["config.target"] == t) & (data["Model"] == "E8")][
                error_metric
            ].mean()
            _std = data[data["config.target"] == t][error_metric].std()
            data.loc[data["config.target"] == t, error_metric] = (
                data[data["config.target"] == t][error_metric] - _mean
            ) / _std
        ylabel = human_labels(error_metric) + " (zscore around mean=E8)"
    else:
        raise NotImplementedError

    return data, ylabel

def plot_acc_vs_speed(
        _df, 
        x,
        y,
        target=None,
        xlabel=None,
        ylabel=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        title="Error vs Time",
        shapestyle=None, # None, config.num_layers
        fname=False,
        palette=cdict,
        hue="Class",
        ls="",
    ):
    data = _df.copy()

    if target is not None:
        data = data[data["config.target"] == target]

    if shapestyle is not None:
        shapekwargs = {
            "shape": shapestyle,
            "markers": marks[: len(list(data[shapestyle].unique()))]
        }
        data[shapekwargs].rename(columns={shapestyle: human_labels(shapestyle)}, inplace=True)
    else:
        shapekwargs = {"marker": 'o'}

    # plot
    reset_plot_styles()
    set_seaborn_style()
    fig, ax = plt.subplots()
    # sns.scatterplot(
    sns.lineplot(
        x=x,
        y=y,
        hue=hue,
        data=data,
        ax=ax,
        palette=palette,
        ls=ls,
        **shapekwargs,
    )
    plt.legend()
    plt.xlabel(xlabel if xlabel is not None else human_labels(x))
    plt.ylabel(ylabel if ylabel is not None else human_labels(y))

    if xmin is not None:
        plt.xlim(left=xmin)
    if xmax is not None:
        plt.xlim(right=xmax)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if ymax is not None:
        plt.ylim(top=ymax)

    if title is not None:
        plt.title(title)

    # vertical xtick labels
    # plt.xticks(rotation=90)

    # horizontal grid
    plt.grid(axis="y")
    set_style_after(ax=ax)

    # save
    if fname:
        plt.savefig(f"{plotfolder}/{fname}.png")
        print(f"Saved plot to \n {plotfolder}/{fname}.png")

    return fig, ax

def plot_acc_vs_speed_errorbar(
        _df, 
        x,
        y,
        target=None,
        xlabel=None,
        ylabel=None,
        title="Error vs Time",
        errbar="sd", # std, sem, ci
        hue="Class",
        ymin=0,
        ymax=None,
        markershape=None, # None, config.num_layers
        fname=False,
        palette=cdict,
    ):
    """_summary_

    Args:
        _df (_type_): normed but not averaged df
    """
    data = _df.copy()

    if target is not None:
        data = data[data["config.target"] == target]


    data = data[["Model", x, y]]

    # new df with error_metric mean 
    # observed only applies if any of the groupers are Categoricals. If True: only show observed values for categorical groupers. If False: show all values for categorical groupers
    _mean = data.groupby(["Model"], observed=False).mean().reset_index()
    # columns and naming
    _mean["Class"] = _mean["Model"].apply(lambda x: "DEQ" if "DEQ" in x else "E")
    _mean["num_layers"] = _mean["Model"].apply(lambda x: int(x[-1]))
    _mean = _mean.sort_values(by=["Class", "Model"], ascending=[False, True])

    if errbar is not None:
        if errbar == "std":
            _err = data.groupby(["Model"]).std().reset_index()
        elif errbar == "sem":
            # unbiased standard error of the mean
            _err = data.groupby(["Model"]).sem().reset_index()
        elif errbar == "ci":
            # compute 90% confidence interval
            _err = data.groupby(["Model"]).agg(lambda x: x.quantile(0.95) - x.mean()).reset_index()
        elif errbar == "sd":
            _err = data.groupby(["Model"]).std().reset_index()
        else:
            raise ValueError(f"Unknown errbar {errbar}")
        _err["Class"] = _err["Model"].apply(lambda x: "DEQ" if "DEQ" in x else "E")
        _err["num_layers"] = _err["Model"].apply(lambda x: int(x[-1]))
        _err = _err.sort_values(by=["Class", "Model"], ascending=[False, True])

    if markershape is not None:
        # make pretty
        _mean.rename(columns={markershape: human_labels(markershape)}, inplace=True)
        if errbar is not None: 
            _err.rename(columns={markershape: human_labels(markershape)}, inplace=True)
        markershape = human_labels(markershape)
        # set markers
        shapekwargs = {
            "style": markershape,
            "markers": marks[: len(list(_mean[markershape].unique()))]
        }
    else:
        shapekwargs = {}

    # plot
    reset_plot_styles()
    set_seaborn_style(palette=PALETTE)
    fig, ax = plt.subplots()
    # sns.pointplot(
    #     x=x,
    #     y=y,
    #     hue="Class",
    #     data=data,
    #     ax=ax,
    #     errorbar="sd", #  “ci”, “pi”, “se”, or “sd"
    #     native_scale=True,
    #     # markers=["o", "x"],
    #     # linestyles=["-", "--"],
    #     # palette=cdict
    # )

    # scatterplot with mean of error_metric and time_metric
    sns.scatterplot(
        x=x,
        y=y,
        hue=hue,
        data=_mean,
        ax=ax,
        palette=palette,
        **shapekwargs,
    )

    if errbar is not None:
        # colors from current sns palette
        colors = sns.color_palette()

        categories = _mean[hue].unique()
        # sort to have E1, E4, E8, DEQ1, DEQ2
        categories = sorted(categories, reverse=True)
        # categories = sorted(categories, key=lambda x: int(x[-1]))

        # plot errorbars one hue at a time
        for _class, _color in zip(categories, colors):
            _data = _mean[_mean[hue] == _class]
            _err_data = _err[_err[hue] == _class]
            if palette is not None:
                _color = palette[_class]
            ax.errorbar(
                x=_data[x],
                y=_data[y],
                xerr=_err_data[x],
                yerr=_err_data[y],
                fmt="none", # none, o
                color=_color,
                elinewidth=1,
                capsize=3,
                capthick=1,
            )

    plt.legend()
    plt.xlabel(xlabel if xlabel is not None else human_labels(x))
    plt.ylabel(ylabel if ylabel is not None else human_labels(y))

    if ymin is not None:
        plt.ylim(bottom=ymin)
    if ymax is not None:
        plt.ylim(top=ymax)

    if title is not None:
        plt.title(title)

    # vertical xtick labels
    # plt.xticks(rotation=90)

    # horizontal grid
    plt.grid(axis="y")
    set_style_after(ax=ax)

    # save
    if fname:
        plt.savefig(f"{plotfolder}/{fname}.png")
        print(f"Saved plot to \n {plotfolder}/{fname}.png")

    return fig, ax

def prep_df_for_table(_df, error_metric="summary.test_f_mae"):
    if "config.model.num_layers" in _df.columns:
        _df["Layers"] = _df["config.model.num_layers"]
    if "config.target" in _df.columns:
        _df["Target"] = _df["config.target"]

    _df["Class"] = _df["config.model_is_deq"].apply(
        lambda x: "Equiformer" if not x else "DEQ"
    )

    _df["Model"] = _df["Class"] + " (" + _df["Layers"].astype(str) + " layers)"
    _df["Model"] = _df["Model"].str.replace("1 layers", "1 layer")
    _df["Model"] = _df["Model"].str.replace("DEQ", "DEQuiformer")

    # cast test_f_mae and test_e_mae to float
    _df[error_metric] = _df[error_metric].astype(float)

    # for row in [error_metric, "test_e_mae"]:
    # print(_df.pivot(index="Model", columns="Target", values=row).to_latex(float_format="%.2f"))

    _df = _df.sort_values(
        by=["Target", "Class", "Layers"], ascending=[True, False, True]
    )
    _df = _df.sort_values(
        by=["Class", "Target", "Layers"], ascending=[False, True, True]
    )

    # reset index
    _df = _df.reset_index(drop=True)

    return _df


def print_table_avg_seeds(
    _df,
    mode="Force",
    add_nfe=False,
    compare_pairwise=False,  # Todo replace with mark_sota
    error_metric="summary.test_f_mae",
):
    """
    format
    cols: Aspirin & Benzene & Ethanol & Malonaldehyde & Naphthalene & Salicylic acid & Toluene & Uracil
    cols: energy & forces
    rows: Equiformer 1 layer & Equiformer 4 layers & Equiformer 8 layers & DEQ 1 layer & DEQ 2 layers
    mean \pm std
    """
    assert mode in [
        "Force",
        "Energy",
        "Time",
    ], f"mode={mode} not in ['Force', 'Energy', 'Time']"

    _df = prep_df_for_table(_df, error_metric=error_metric)

    # print(f'\n{mode} df:\n', _df[["Model", "Class", "Layers", "Target", "seed", error_metric, "test_e_mae"]])

    # mean
    # dfmean = _df.groupby(["Model", "Target"]).mean(numeric_only=True).reset_index()
    # print(f'\n{mode} dfmean:\n', dfmean[["Model", "Target", error_metric, "test_e_mae"]])

    # padding of 9 chars to compensate for 'mathbf{}'
    padding = " " * 9

    print(f"\n{mode} table:")
    first_deq = True
    lines = []
    nfe_lines = []
    mean_values = np.zeros((len(_df["Model"].unique()), len(_df["Target"].unique())))

    for _r, row in enumerate(list(_df["Model"].unique())):
        line = [row + " & "]
        nfe_line = []
        # print(_df[_df["Model"] == row].pivot(index="Target", columns="Model", values=error_metric).to_latex(float_format="%.2f"))
        for _c, col in enumerate(list(_df["Target"].unique())):
            val = _df[(_df["Model"] == row) & (_df["Target"] == col)][
                error_metric
            ].values
            seeds = _df[(_df["Model"] == row) & (_df["Target"] == col)]["seed"].values
            nfe_val = _df[(_df["Model"] == row) & (_df["Target"] == col)]["NFE"].values
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

        line[-1] = line[-1][:-2] + "\\\\"  # NFE has to come first
        line[0] = line[0].replace("Equiformer", "\equiformer{}")
        if "DEQ" in row and first_deq == True:
            # print("\hline")
            first_deq = _r
        lines.append(line)
        nfe_lines.append(nfe_line)

    # mark the best row in each column
    for _c in range(mean_values.shape[1]):
        if mode == "Time":
            if compare_pairwise:
                # only compare 1-layer DEQ vs 4-layer Equiformer,
                # and 2-layer DEQ vs 8-layer Equiformer
                # ingore the first row: Equiformer 1 layer
                # compare the second row (Equiformer 4 layers) and the fourth row (DEQ 2 layers)
                for pair in [(1, 3), (2, 4)]:
                    _means = mean_values[:, _c]
                    # shift all errors up, except the pair
                    mask = np.ones(_means.shape, dtype=bool) * 1000.0
                    mask[pair[0]] = 0
                    mask[pair[1]] = 0
                    _means = _means + mask
                    best_row = np.argmin(_means)
                    # the first column in each line is the row name
                    line_prev = lines[best_row][_c + 1]
                    line_prev = (
                        line_prev.replace("$", "")
                        .replace(" &", "")
                        .replace("\\\\", "")
                        .replace(padding, "")
                    )
                    lines[best_row][_c + 1] = "$ \\mathbf{" + line_prev + "} $"
                    if _c == mean_values.shape[1] - 1:
                        lines[best_row][_c + 1] += "\\\\"
                    else:
                        lines[best_row][_c + 1] += " &"

            else:
                _means = mean_values[:, _c]
                best_row = np.argmin(_means)
                # the first column in each line is the row name
                line_prev = lines[best_row][_c + 1]
                line_prev = (
                    line_prev.replace("$", "")
                    .replace(" &", "")
                    .replace("\\\\", "")
                    .replace(padding, "")
                )
                lines[best_row][_c + 1] = "$ \\mathbf{" + line_prev + "} $"
                if _c == mean_values.shape[1] - 1:
                    lines[best_row][_c + 1] += "\\\\"
                else:
                    lines[best_row][_c + 1] += " &"

        else:
            # ?
            best_row = np.argmin(mean_values[:, _c])
            # lines first column is the row name
            line_prev = lines[best_row][_c + 1]
            line_prev = (
                line_prev.replace("$", "")
                .replace(" &", "")
                .replace("\\\\", "")
                .replace(padding, "")
            )
            lines[best_row][_c + 1] = "$ \\mathbf{" + line_prev + "} $"
            if _c == mean_values.shape[1] - 1:
                lines[best_row][_c + 1] += "\\\\"
            else:
                lines[best_row][_c + 1] += " &"

    # add in the NFE at each cell
    if mode == "Time" and add_nfe:
        for _r, line in enumerate(nfe_lines):
            for _c, cell in enumerate(line):
                lines[_r][_c + 1] = cell + lines[_r][_c + 1]

    print(f"\n{mode} table:")
    for _l, line in enumerate(lines):
        if _l == first_deq:
            print("\midrule[0.6pt]")
        print("".join(line))

    return


def print_table_time_forces_avg_seeds(
    _df,
    compare_pairwise=False,
    error_metric="summary.test_f_mae",
    # energy_metric="summary.test_e_mae",
):

    _df = prep_df_for_table(_df, error_metric=error_metric)

    # padding of 9 chars to compensate for 'mathbf{}'
    padding = " " * 9

    print(f"\nCombined Force+Time table:")
    lines_both = []
    for mode in ["Force", "Time"]:
        first_deq = True
        lines = []
        mean_values = np.zeros(
            (len(_df["Model"].unique()), len(_df["Target"].unique()))
        )
        if mode == "Force":
            metric = error_metric
        else:  # Time
            metric = "time_forward_per_batch_test_lowest"
        for _r, row in enumerate(list(_df["Model"].unique())):
            line = [row + " & "]
            # print(_df[_df["Model"] == row].pivot(index="Target", columns="Model", values=error_metric).to_latex(float_format="%.2f"))
            for _c, col in enumerate(list(_df["Target"].unique())):
                val = _df[(_df["Model"] == row) & (_df["Target"] == col)][metric].values
                seeds = _df[(_df["Model"] == row) & (_df["Target"] == col)][
                    "seed"
                ].values
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

            line[-1] = line[-1][:-2] + "\\\\"  # NFE has to come first
            line[0] = line[0].replace("Equiformer", "\equiformer{}")
            if "DEQ" in row and first_deq == True:
                # print("\hline")
                first_deq = _r
            lines.append(line)

        # mark the best row in each column
        for _c in range(mean_values.shape[1]):
            if mode == "Time":
                if compare_pairwise:
                    # ingore the first row: Equiformer 1 layer
                    # compare the second row (Equiformer 4 layers) and the fourth row (DEQ 2 layers)
                    for pair in [(1, 3), (2, 4)]:
                        _means = mean_values[:, _c]
                        mask = np.ones(_means.shape, dtype=bool) * 1000.0
                        mask[pair[0]] = 0
                        mask[pair[1]] = 0
                        _means = _means + mask
                        best_row = np.argmin(_means)
                        # lines first column is the row name
                        line_prev = lines[best_row][_c + 1]
                        line_prev = (
                            line_prev.replace("$", "")
                            .replace(" &", "")
                            .replace("\\\\", "")
                            .replace(padding, "")
                        )
                        lines[best_row][_c + 1] = "$ \\mathbf{" + line_prev + "} $"
                        if _c == mean_values.shape[1] - 1:
                            lines[best_row][_c + 1] += "\\\\"
                        else:
                            lines[best_row][_c + 1] += " &"
                else:
                    _means = mean_values[:, _c]
                    best_row = np.argmin(_means)
                    # lines first column is the row name
                    line_prev = lines[best_row][_c + 1]
                    line_prev = (
                        line_prev.replace("$", "")
                        .replace(" &", "")
                        .replace("\\\\", "")
                        .replace(padding, "")
                    )
                    lines[best_row][_c + 1] = "$ \\mathbf{" + line_prev + "} $"
                    if _c == mean_values.shape[1] - 1:
                        lines[best_row][_c + 1] += "\\\\"
                    else:
                        lines[best_row][_c + 1] += " &"
            else:
                best_row = np.argmin(mean_values[:, _c])
                # lines first column is the row name
                line_prev = lines[best_row][_c + 1]
                line_prev = (
                    line_prev.replace("$", "")
                    .replace(" &", "")
                    .replace("\\\\", "")
                    .replace(padding, "")
                )
                lines[best_row][_c + 1] = "$ \\mathbf{" + line_prev + "} $"
                if _c == mean_values.shape[1] - 1:
                    lines[best_row][_c + 1] += "\\\\"
                else:
                    lines[best_row][_c + 1] += " &"

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
            lines[_r][_c] = (
                lines_force[_r][_c].replace("\\\\", "&") + lines_time[_r][_c]
            )

    # compute relative speedup between Equiformer 8 layers and DEQ 2 layers
    avg_speedup = 0.0
    for _c, col in enumerate(list(_df["Target"].unique())):
        val_eq8 = _df[
            (_df["Model"] == "Equiformer (8 layers)") & (_df["Target"] == col)
        ]["time_forward_per_batch_test_lowest"].values
        val_deq2 = _df[
            (_df["Model"] == "DEQuiformer (2 layers)") & (_df["Target"] == col)
        ]["time_forward_per_batch_test_lowest"].values
        if len(val_eq8) == 0:
            print(f" Warning: No value for Equiformer 8 layers for {col}")
            continue
        if len(val_deq2) == 0:
            print(f" Warning: No value for DEQ 2 layers for {col}")
            continue
        for seed in range(1, 4):
            try:
                speedup = val_eq8[seed] / val_deq2[seed]
                print(f" Target={col} Speedup: {speedup:.2f}")
            except:
                pass
        speedup = val_eq8[0] / val_deq2[0]
        # speedup = val_deq2[0] / val_eq8[0]
        print(f"Target={col} Speedup: {speedup:.2f}")
        avg_speedup += speedup
    avg_speedup /= len(list(_df["Target"].unique()))
    print(f"Average speedup: {avg_speedup:.2f}")

    print(f"\nCombined Force+Time table:")
    for _l, line in enumerate(lines):
        if _l == first_deq:
            print("\midrule[0.6pt]")
        print("".join(line))

def get_keys_history(run_id, project=projectmd,
        keys=None, download=False, sname=None,
        allcols=True,
        # save_plot=False, logscale=False, ymax=None, xmax=None,
    ):
    if keys is None:
        if project == projectmd:
            keys = ["train_f_mae", "test_f_mae", "val_f_mae"]
        elif project == projectmd:
            keys = ["train/forces_mae", "val/forces_mae"]

    api = wandb.Api()
    run = api.run(project + "/" + run_id)
    run_name = run.name
    print("\nrun_id:", run_id)
    print("name:  ", run.name)
    mname = "".join(e for e in run_name if e.isalnum())
    # save as parquet
    sname = (
        f"{plotfolder}/traintestfmae_{run_id}_{mname}.parquet"
    )
    # try to load from csv
    if download == False:
        try:
            df = pd.read_csv(sname)
            print(f"Loaded from csv: {sname}")

        except FileNotFoundError:
            download = True
            print(f"File not found: {sname}")
    if download:
        print("Downloading run history...")
        if allcols:
            history = run.scan_history()
        else:
            history = run.history(keys=keys + ["_step", "_runtime", "epoch"])

        # https://github.com/wandb/wandb/blob/v0.18.0/wandb/apis/public/history.py#L80
        print(f'History: {type(history)}')

        # turn history into dataframe
        df = pd.DataFrame(history)
        # rename artifact_name to error_type
        # df = df.rename(columns={artifact_name: col})
        print(f'len(df) of downloaded history: {len(df)}')
        # print('columns:', df.columns)

        # drop rows that are None
        if keys is not None:
            df = df.dropna(subset=keys, how='all')
            print(f'len(df) after dropping na: {len(df)}')

        # # if artifact is a list
        # # turn _step into a list of same length as trace
        # length = len(df[col].iloc[0])
        # df["train_step"] = df["_step"].apply(lambda x: [x] * length)

        # # list_step is a list [0, 1, 2, ...]
        # # df["list_step"] = df.index
        # df["list_step"] = df[col].apply(lambda x: list(range(len(x))))
        # # print('df: \n', df.head())

        # # flatten
        # df = df.explode(["train_step", "list_step", col])
        # print(f'len(df) after flatten: {len(df)}')
        
        # print('df: \n', df.head())

        # test_fpr_f_mae
        cols_to_keep = keys + [
            "epoch", "_step", "_runtime", 
            "avg_n_fsolver_steps_val", "avg_n_fsolver_steps_test", "avg_n_fsolver_steps_train",
        ]

        df["run_id"] = run_id
        df["run_name"] = run.name
        df["Model"] = ("E" if not run.config["model_is_deq"] else "DEQ") + str(run.config["model"]["num_layers"])
        df["Class"] = "E" if not run.config["model_is_deq"] else "DEQ"
        df["model.num_layers"] = run.config["model"]["num_layers"]

        # save dataframe using runname
        df.to_csv(sname)
    
    return df

def plot_loss_curve(
        df, x="_step", y="test_f_mae", 
        xlabel=None, ylabel=None, logscale=False, 
        ymax=None, xmax=None, 
        ymin=None, xmin=None,
        fname=None, title=None, hue="Class",
        palette=cdict,
):
    _df = df.copy()
    _df.dropna(subset=[y], inplace=True)
    if xmin is not None:
        _df = _df[_df[x] >= xmin]
    else:
        xmin = _df[x].min()
    set_seaborn_style()

    fig, ax = plt.subplots()
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
    # plt.gca().ticklabel_format(useMathText=True)

    # plot: x=solver_step, y=error_type, hue=train_step
    sns.lineplot(
        data=_df,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        palette=palette,
    )
    plt.xlabel(human_labels(x) if xlabel is None else xlabel)
    plt.ylabel(human_labels(y) if ylabel is None else ylabel)
    if logscale:
        plt.yscale("log")
    if ymax is not None:
        # cant plot 0 on logscale
        # plt.ylim(1e-12, ymax)
        plt.ylim(top=ymax)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if xmin is not None:
        plt.xlim(left=xmin)
    else:
        plt.xlim(left=0)
    if xmax is not None:
        plt.xlim(right=xmax)
    # legend title
    # plt.title(f"{run_name}")
    if title is not None:
        plt.title(title)

    set_style_after(ax) # , fs=10

    # legend outside the plot on the right
    # plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)
    # ax.get_legend().get_frame().set_linewidth(0.0)

    if fname is not None:
        fname = f"{plotfolder}/{fname}.png"
        plt.savefig(fname)
        print(f"Saved plot to \n {fname}")

    return fig, ax

def get_runs_status_from_wandb(
    project=projectmd,
    filters={
        # "tags": "inference2",
        # "$and": [{"tags": "md17"}, {"tags": "eval"}],
        # "state": "finished",
        # "$or": [{"tags": "md17"}, {"tags": "main2"}, {"tags": "inference"}],
        # "state": "crashed",
        "$or": [{"state": "failed"}, {"state": "crashed"}],
    },
    fname="",
    hours_since=48,
):
    """Download runs from wandb."""
    # hosts, hostname = ["tacozoid11", "tacozoid10"], "taco"
    fname = f"runs_p_{project.replace('project', '')}" + fname
    fullfname = f"{plotfolder}/{fname}.csv"

    api = wandb.Api()
    runs = api.runs(
        project,
        filters,
    )
    run_ids = [run.id for run in runs]
    # print(f"Found {len(run_ids)} runs:")

    infos_acc = []
    for run in runs:
        # filters
        # host = requests.get(run.file("wandb-metadata.json").url).json()["host"]
        # if host not in hosts:
        #     print(f"Skipping run {run.id} {run.name} because of host={host}")
        #     continue

        # print([
        #     k for k in run.summary.keys() 
        #     if 
        #     k.startswith("_")
        #     # (not "gradient" in k) 
        #     # and (not "config" in k)
        #     # and (not "summary" in k)
        # ])

        # print(run.summary["_timestamp"])
        # print(run.summary["_runtime"])
        # print(run.summary["_step"])

        # timestep is of format 1722367178.5828128
        # convert to datetime
        if "_timestamp" not in run.summary:
            # print(f"Skipping run {run.id} {run.name} because of missing _timestamp")
            # runs never started, failed before logging the first step
            continue
        timestemp = run.summary["_timestamp"]
        date = datetime.datetime.fromtimestamp(timestemp)
        # remove microseconds
        # date = date.strftime("%Y-%m-%d %H:%M:%S")

        # filter if run is older than .. hours
        if datetime.datetime.now() - date > datetime.timedelta(hours=hours_since):
            # print(f"Skipping run {run.id} {run.name} because of time")
            continue
        
        # runtime is of format 215989.28339982033
        # convert to hours
        hours = run.summary["_runtime"] / 3600

        info = {
            "run_id": run.id,
            "run_name": run.name,
            "date": date,
            "hours": hours,
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
        
        # get log
        # file = run.file('output.log')
        # file.download() # root="."
        # file.display(height=420, hidden=(False))

        # print the name
        print(run.name)
        print("", info['config.slurm_job_id'])
        if project == projectmd:
            print(f" epoch={info['summary.epoch']} / {info['config.epochs']}")
        elif project == projectoc:
            print(f" epoch={info['summary.epoch']} / {info['config.optim.max_epochs']}")
        # print(f" date={date}")
        # date as day hour:min
        print(f" date={date.strftime('%d %H:%M')}")
        print(f" hours={hours:.2f}")

        # metadata
        # host = requests.get(run.file("wandb-metadata.json").url).json()["host"]
        # info["host"] = host

        infos_acc.append(info)

    df = pd.DataFrame(infos_acc)


    return df

def print_restart_commands(
        _df, project=projectmd,
        ignore_overrides=["wandb_tags"],
    ):
    if project == projectmd:
        prefix = "launchrun"
        tag = "['speedmd_v1']"
    elif project == projectoc:
        prefix = "launchoc"
        tag = "['speedoc_v1']"

    # ignore_overrides += ["config.target"]
    
    # df[["config.override_dirname"]]
    # print(" ")
    _num_runs = 0
    for i, row in _df.iterrows():
        overrides = row["config.override_dirname"].split(",")
        _overrides = []
        for o in overrides:
            _add = True
            for so in ignore_overrides:
                if so in o:
                    _add = False
                    break
                
            if _add:
                _overrides.append(o)
        overrides = " ".join([f"{o}" for o in _overrides])
        
        # if "config.target" in row:
        if project == projectmd:
            target = " target=" + row["config.target"]
        else:
            target = ""
        print(
            f"{prefix} {overrides}",
            # f"{prefix} {overrides}{target}",
            # f"+inf=bs1 +deq_kwargs_fpr.f_tol=1e-1 wandb_tags={tag}",
        )