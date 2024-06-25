import py3Dmol

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import wandb
import omegaconf
import copy
import os, sys, pathlib
import pickle
import scipy

from torch_geometric.loader.dataloader import Collater


project = "EquilibriumEquiFormer"

from src.deq2ff.plotting.style import set_seaborn_style, set_style_after

# parent folder of the plot
plotfolder = "/ssd/gen/equilibrium-forcefields/src/deq2ff/plotting/"
plotfolder = os.path.join(plotfolder, "plots")

chemical_symbols = [
    "_",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
]

# nstep, nstep_std, nstep_max, nstep_min, f_mae, fpabs, fprel
labels_human = {
    "f_mae": "Force MAE",
    "nstep": "Solver Steps",
    "nstep_min": r"$min_{n \in mol}\ SolverSteps_{n}$",
    "nstep_max": r"$max_{n \in mol}\ SolverSteps_{n}$",
    "nstep_std": r"$std_{n \in mol}\ SolverSteps_{n}$",
    "fpabs": r"Abs FP Error $|f(z^*) - z^*|$",
    "fprel": r"Rel FP Error $|f(z^*) - z^*| / |z^*|$",
}


def plot_luca():

    # Function to parse the extended .xyz file
    def parse_xyz(file_content):
        lines = file_content.strip().split("\n")
        energy = float(lines[1])
        atoms = []
        positions = []
        forces = []

        for line in lines[2:]:
            parts = line.split()
            atom = parts[0]
            position = list(map(float, parts[1:4]))
            force = list(map(float, parts[4:7]))
            atoms.append(atom)
            positions.append(position)
            forces.append(force)

        return energy, atoms, positions, forces

    xyz_content = """21
    -406757.5912640221
    C 2.2393 -0.3791 0.263 -1.8342235724199345 1.0243422546435585 -2.2543675968818286
    C 0.8424 1.9231 -0.4249 -5.591218505993106 3.309947916554474 -4.351241814202116
    C 2.8709 0.8456 0.2722 10.418566672065914 1.8167662124620771 3.0534166566152257
    C 2.1751 1.9935 -0.0703 4.368071217417887 2.724674552067337 -0.6607672787856174
    C -3.4838 0.4953 -0.0896 12.734432897968423 6.150024154554914 -22.55665960891591
    C 0.891 -0.4647 -0.0939 17.07562357556979 -21.519462409575137 6.482720010705403
    C 0.1908 0.6991 -0.4402 17.672138859460603 -2.5279522871197924 10.232899560602233
    O -0.9633 -1.8425 -0.4185 -21.761366282279926 -13.004609632046286 -3.5406889760393843
    O -1.6531 0.8889 1.3406 6.739812810181667 3.7175695263667086 24.810812505027446
    O 0.8857 -2.8883 0.2267 56.17724192114808 12.013353747302904 12.981529441068616
    C 0.209 -1.772 -0.1069 -36.624724852586375 -9.130682491313335 -7.689925680840504
    C -2.0185 0.6853 0.2071 -19.294461082691402 2.3522342221469987 39.64692577448302
    O -1.1189 0.6285 -0.7886 -22.633992461721434 9.279295816206334 -48.17346319307507
    H 0.3962 -3.7219 0.2035 -20.280029988378665 4.8815280035211055 -6.036836215230837
    H 2.7867 -1.2719 0.5268 1.4100065194139182 -7.83443201508828 1.5333644164256273
    H 0.3069 2.8224 -0.6911 -7.049602938267278 5.1625303286628315 -1.6637621588358686
    H 3.913 0.9108 0.5482 8.59088822864889 0.6221194865601325 2.4642140511070694
    H 2.6781 2.9492 -0.0604 3.4005013876011967 8.230906774895088 -0.0026730380257048757
    H -3.736 -0.5623 -0.012 0.3959199762841433 -6.127246351237042 -1.133668533722564
    H -4.0763 1.0637 0.6273 2.0821138935464574 0.9967566887042344 4.038063882192225
    H -3.6988 0.8471 -1.0986 -5.995698274968782 -2.137664498268801 -7.179892203671586
    """

    # Parse the .xyz file content
    energy, atoms, positions, forces = parse_xyz(xyz_content)
    print("atoms", atoms)

    # Generate XYZ string for py3Dmol
    xyz_str = f"{len(atoms)}\n\n"
    for atom, pos in zip(atoms, positions):
        xyz_str += f"{atom} {pos[0]} {pos[1]} {pos[2]}\n"

    # Create the 3D visualization
    view = py3Dmol.view(width=800, height=600)
    view.addModel(xyz_str, "xyz")

    scaling_factor = 0.05  # Adjust this value as necessary
    for pos, force in zip(positions, forces):
        # print('pos', pos)
        # print('force', force)
        start = {"x": pos[0], "y": pos[1], "z": pos[2]}
        end = {
            "x": pos[0] + force[0] * scaling_factor,
            "y": pos[1] + force[1] * scaling_factor,
            "z": pos[2] + force[2] * scaling_factor,
        }
        view.addArrow({"start": start, "end": end, "radius": 0.08, "color": "orange"})

    style = {"stick": {"radius": 0.2}, "sphere": {"scale": 0.2}}
    view.setStyle({"model": -1}, style, viewer=None)
    view.zoomTo()
    view.show()


# plot_luca()


def get_data(run_id, datasplit="test_fpreuse"):
    # idx_table
    table_key = f"idx_table_{datasplit}"
    # table_key = "_idx_table_{_datasplit}"

    # a = artifacts[-1]
    # table = a.get(table_key)
    # df = pd.DataFrame(table)

    cwd = os.getcwd()
    filename = cwd + "/plot_loss_per_idx/"
    filename += f"run-{run_id}-{table_key}.pkl"
    print(f"filename: {filename}")

    api = wandb.Api()

    # get run
    run = api.run(f"{project}/{run_id}")
    data_path = run.config["data_path"]
    target = run.config["target"]
    dname = run.config["dname"]
    train_size = run.config["train_size"]
    val_size = run.config["val_size"]
    test_size = run.config["test_size"]
    seed = run.config["seed"]
    args_datasplit = run.config["datasplit"]
    fpreuse_test = run.config["fpreuse_test"]
    contrastive_loss = run.config["contrastive_loss"]
    deq = run.config["model_is_deq"]


    # load the data
    import equiformer.datasets.pyg.md_all as md_all

    print(f"Loading dataset...")
    (
        train_dataset,
        val_dataset,
        test_dataset,
        test_dataset_full,
        all_dataset,
    ) = md_all.get_md_datasets(
        root=data_path,
        dataset_arg=target,
        dname=dname,
        train_size=train_size,
        val_size=val_size,
        test_size=None,  # influences data splitting
        test_size_select=test_size,  # doesn't influence data splitting
        seed=seed,
        order="consecutive_all",
    )

    metrics = [
        "norm_mean",
        "norm",
        "cos_mean",
        "cos",
        "fnorm",
        "fnorm_mean",
        "fnorm_max",
        "max",
        "fnorm_std",
    ]
    if os.path.exists(filename):
        print(f"Loading data from {filename}")
        with open(filename, "rb") as f:
            df = pickle.load(f)

    else:
        a = api.artifact(f"{project}/run-{run_id}-{table_key}:latest")
        # apath = a.download()
        table = a.get(table_key)
        df = pd.DataFrame(data=table.data, columns=table.columns)
        print(f"Found data: {df.shape}")

        print(f"Adding forces...")
        # add forces column to the dataframe from data.dy
        collate = Collater(None, None)
        dataset = all_dataset
        df["forces"] = [None] * df.shape[0]
        df["z"] = [None] * df.shape[0]
        for i, row in df.iterrows():
            idx = int(row["idx"])
            data = collate([dataset[idx]])
            f = data.dy  # torch.Tensor
            f = f.detach().cpu().numpy()
            df.at[i, "forces"] = f
            # add z to the dataframe
            z = data.z
            z = [_z.item() for _z in z]
            df.at[i, "z"] = z
        
        # add DEQ or Equ to the dataframe in the "DEQ" column
        df["Model"] = ["DEQ" if deq else "Equ"] * df.shape[0]
        df["target"] = [target] * df.shape[0]

        print(f"Calculating f_delta between steps...")
        # from forces get the force delta between consecutive steps
        # df["f_delta"] = df["forces"].diff().mean()
        for metric in metrics:
            df["f_delta_" + metric] = [-1.0] * df.shape[0]
        for i, row in df.iterrows():
            f = row["forces"]
            if i == 0:
                continue
            try:
                f_prev = df.at[i - 1, "forces"]  # num_atoms x 3
            except Exception as e:
                print(f"Error at idx {i}.")
                print(f"df:\n{df.iloc[i-1]}")
                raise e

            for metric in metrics:
                # f: [num_atoms x 3]
                # f - f_prev: [num_atoms x 3]
                # np.linalg.norm(f - f_prev, axis=1): [num_atoms]
                # np.linalg.norm(f - f_prev, axis=1).mean(): scalar
                if metric == "norm_mean":
                    f_delta = np.linalg.norm(f - f_prev, axis=1).mean()
                    xlabel = r"$\frac{1}{N} \sum_{n \in atoms}^{N}|Force_t^{(n)} - Force_{t-1}^{(n)}|_2$"
                elif metric == "norm":
                    f_delta = np.linalg.norm(f - f_prev)
                    xlabel = r"$|Force_t^{(1\cdots n)} - Force_{t-1}^{(1\cdots n)}|_2$"
                elif metric == "cos_mean":

                    def cosine_similarity(x, y):
                        return np.sum(x * y, axis=1) / (
                            np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
                        )

                    f_delta = np.mean(cosine_similarity(f, f_prev))
                    xlabel = r"$\frac{1}{N} \sum_{n \in atoms}^{N} \cos(Force_t^{(n)}, Force_{t-1}^{(n)})$"
                elif metric == "cos":

                    def cosine_similarity(x, y):
                        return np.sum(x * y) / (np.linalg.norm(x) * np.linalg.norm(y))

                    f_delta = cosine_similarity(f, f_prev)
                    xlabel = r"$\cos(Force_t^{(1\cdots n)}, Force_{t-1}^{(1\cdots n)})$"
                elif metric == "max":
                    f_delta = np.linalg.norm(f - f_prev, axis=1).max()
                    xlabel = (
                        r"$\max_{n \in atoms}|Force_t^{(n)} - Force_{t-1}^{(n)}|_2$"
                    )
                # not really delta
                elif metric == "fnorm":
                    f_delta = np.linalg.norm(f)
                elif metric == "fnorm_mean":
                    f_delta = np.linalg.norm(f, axis=1).mean()
                    xlabel = r"$\frac{1}{N} \sum_{n \in atoms}^{N}|Force_t^{(n)}|_2$"
                elif metric == "fnorm_max":
                    f_delta = np.linalg.norm(f, axis=1).max()
                    xlabel = r"$\max_{n \in atoms}|Force_t^{(n)}|_2$"
                elif metric == "fnorm_std":
                    f_delta = np.linalg.norm(f, axis=1).std()
                    xlabel = r"$\sigma_{n \in atoms}|Force_t^{(n)}|_2$"
                else:
                    raise ValueError(f"metric {metric} not known.")

                df.at[i, "f_delta_" + metric] = float(f_delta)
                assert np.allclose(
                    df.at[i, "z"], df.at[i - 1, "z"]
                ), f"z not equal at idx {i} and {i-1}: {df.at[i, 'z']} and {df.at[i-1, 'z']}"

        # save the dataframe
        # create the folder if it doesn't exist
        with open(filename, "wb") as f:
            pickle.dump(df, f)

    # cast to float64
    # for metric in metrics:
    #     df["f_delta_"+metric].astype('float64', copy=False)
    # df["nstep"].astype('float64', copy=False)
    # print("df dtypes", df.dtypes)

    print(f"Removing non-fpreuse rows...")
    # remove every patch_size row (because there is no fpreuse)
    # df = df.iloc[1:]
    test_size = int(run.config["test_size"])
    test_patches = int(run.config["test_patches"])
    # patch_size = test_size // test_patches
    patch_size = test_size
    print(f"patch_size={patch_size}")

    # remove every patch_size row (because there is no fpreuse)
    dffp = copy.deepcopy(df)
    print(f"len(df) before filtering: {len(dffp)}")
    dffp = dffp.iloc[patch_size - 1 :: patch_size]
    print(f"len(df) fpreuse only    : {len(dffp)}")
    # print(dffp.head())

    return dffp, df, all_dataset


def filter_z_score(_df, col="f_mae", std=3, abs=False, return_outliers=False):
    """_summary_

    Args:
        _df (_type_): _description_
        col (str, optional): _description_. Defaults to "f_mae".
        std (int, optional): How many standard deviations to keep. Defaults to 3.
        abs (bool, optional): If both positive and negative outliers should be removed. Defaults to False.

    Returns:
        _type_: _description_
    """
    before = len(_df)
    if col is None:
        # remove all rows that have outliers in at least one column
        z = scipy.stats.zscore(_df)
        if abs:
            z = np.abs(z)
        if return_outliers:
            mask = (z > std).all(axis=1)
        else:
            mask = (z < std).all(axis=1)
    else:
        # remove outliers with high f_mae
        z = scipy.stats.zscore(_df[col])
        if abs:
            z = np.abs(z)
        if return_outliers:
            mask = z > std
        else:
            mask = z < std
    _df = _df[mask]
    print(f"Removed {before - len(_df)} outliers.")
    return _df


def filter_quantile(_df, col="f_mae", q=0.99):
    # remove largest q% of the data
    before = len(_df)
    qcut = _df[col].quantile(q)
    _df = _df[_df[col] < qcut]
    print(f"Removed {before - len(_df)} outliers.")
    return _df


def filter_quantile_lower_upper(_df, col="f_mae", q=0.99):
    # If one need to remove lower and upper outliers, combine condition with an AND statement:
    before = len(_df)
    q_low = _df[col].quantile(1 - q)
    q_hi = _df[col].quantile(q)
    _df = _df[(_df[col] < q_hi) & (_df[col] > q_low)]
    print(f"Removed {before - len(_df)} outliers.")
    return _df


def plot_model_py3d(idx, dfall, dataset, datasplit, run_id, next=False):
    collate = Collater(None, None)

    # get the data
    # data = collate([dataset[_idx] for _idx in idx])
    data = collate([dataset[idx]])
    # positions = data.pos
    # forces = data.dy
    # e = data.y
    positions = data.pos.tolist()
    forces = data.dy.tolist()
    e = data.y.tolist()

    z = data.z
    # print('z', z)
    atoms = [chemical_symbols[int(_z)] for _z in z]
    # print('atoms', atoms)

    # Generate XYZ string for py3Dmol
    xyz_str = f"{len(atoms)}\n\n"
    for atom, pos in zip(atoms, positions):
        xyz_str += f"{atom} {pos[0]} {pos[1]} {pos[2]}\n"

    # Create the 3D visualization
    view = py3Dmol.view(width=800, height=600)
    view.addModel(xyz_str, "xyz")

    scaling_factor = 0.05  # Adjust this value as necessary
    radius = 0.08

    for pos, force in zip(positions, forces):
        # print('pos', pos)
        # print('force', force)
        start = {"x": pos[0], "y": pos[1], "z": pos[2]}
        end = {
            "x": pos[0] + force[0] * scaling_factor,
            "y": pos[1] + force[1] * scaling_factor,
            "z": pos[2] + force[2] * scaling_factor,
        }
        view.addArrow({"start": start, "end": end, "radius": radius, "color": "orange"})

    if next:
        idx = idx - 1
        data = collate([dataset[idx]])
        pos2 = data.pos.tolist()
        force2 = data.dy.tolist()
        e2 = data.y.tolist()
        z2 = data.z
        atoms2 = [chemical_symbols[int(_z)] for _z in z2]

        # Generate XYZ string for py3Dmol
        xyz_str = f"{len(atoms2)}\n\n"
        for atom, pos in zip(atoms2, pos2):
            xyz_str += f"{atom} {pos[0]} {pos[1]} {pos[2]}\n"

        # Create the 3D visualization
        view.addModel(xyz_str, "xyz")

        for pos, force in zip(pos2, force2):
            start = {"x": pos[0], "y": pos[1], "z": pos[2]}
            end = {
                "x": pos[0] + force[0] * scaling_factor,
                "y": pos[1] + force[1] * scaling_factor,
                "z": pos[2] + force[2] * scaling_factor,
            }
            view.addArrow(
                {"start": start, "end": end, "radius": radius, "color": "blue"}
            )

    style = {"stick": {"radius": 0.1}, "sphere": {"scale": 0.2}}
    view.setStyle({"model": -1}, style, viewer=None)
    view.zoomTo()
    view.show()


def plot_mol_plt(idx, dfall, dataset, datasplit, run_id, side_by_side=False):

    collate = Collater(None, None)

    # get the data
    # data = collate([dataset[_idx] for _idx in idx])
    data = collate([dataset[idx]])
    pos = data.pos
    f = data.dy
    e = data.y
    z = data.z

    quiver_kwargs = dict(
        length=0.02,
        # headwidth=3, headlength=5, headaxislength=4.5,
        # width=.05, headwidth=1, headlength=1, headaxislength=1,
        # 3d
        # arrow_length_ratio=0.1,
    )

    # create a 3d plot of the molecule
    if side_by_side:
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(121, projection="3d")
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="red", label="Atoms")
    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        pos[:, 2],
        f[:, 0],
        f[:, 1],
        f[:, 2],
        color="blue",
        label="Forces",
        **quiver_kwargs,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # add previous idx to the plot
    min_idx_prev = idx - 1
    data = collate([dataset[min_idx_prev]])
    pos_prev = data.pos
    f_prev = data.dy
    e_prev = data.y

    if side_by_side:
        ax2 = fig.add_subplot(122, projection="3d")
    else:
        ax2 = ax
    ax2.scatter(
        pos_prev[:, 0], pos_prev[:, 1], pos_prev[:, 2], c="green", label="Atoms (prev)"
    )
    ax2.quiver(
        pos_prev[:, 0],
        pos_prev[:, 1],
        pos_prev[:, 2],
        f_prev[:, 0],
        f_prev[:, 1],
        f_prev[:, 2],
        color="purple",
        label="Forces (prev)",
        **quiver_kwargs,
    )

    # increase dpi for better quality
    plt.tight_layout()

    # plt.show()
    # save the plot
    plotname = f"molecule_{datasplit}-{run_id}.png"
    plotpath = os.path.join(plotfolder, plotname)
    plt.savefig(plotpath)
    print(f"Saved plot to\n {plotpath}")

    return fig


line_kws = {"lw": 2}


def plot_loss_per_idx(
        dffp, dfall, datasplit, run_id, logy=False, fig=None, alpha=1., xmin=None, xmax=None, ymin=None, ymax=None, style="scatter"
    ):
    """Only considers the fpreuse idxs."""

    # columns: "idx", "e_mae", "f_mae", "nstep"
    # plot the loss per idx
    set_seaborn_style()
    colors = sns.color_palette("dark")

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.gca()
    # sns.lineplot(data=dffp, x="idx", y="e_mae", ax=ax, label="Energy MAE")
    if style == "line":
        sns.lineplot(
            data=dffp,
            x="idx",
            y="f_mae",
            ax=ax,
            # label=f"Force MAE {label}",
            lw=1,
            # c=colors[c],
            palette="dark",
            hue="Model",
            alpha=alpha,
        )
    elif style == "scatter":
        sns.scatterplot(
            data=dffp,
            x="idx",
            y="f_mae",
            ax=ax,
            # label=f"Force MAE {label}",
            # lw=1,
            # c=colors[c],
            palette="dark",
            hue="Model",
            alpha=alpha,
        )
    # ax.set_yscale("log")
    ax.set_xlabel("Index")
    ax.set_ylabel("MAE")
    # ax.legend()
    set_style_after(ax, legend=True)

    if xmin is not None:
        plt.xlim(left=xmin)
    if xmax is not None:
        plt.xlim(right=xmax)
    if ymin is not None:
        plt.ylim(bottom=ymin)
    if ymax is not None:
        plt.ylim(top=ymax)

    if logy:
        ax.set_yscale("log")

    plt.tight_layout()

    plt.title(f"{dffp['target'].iloc[0].capitalize()} (FPreuse only)")

    # save the plot
    # plotname = f"loss_per_idx_{datasplit}-{run_id}.png"
    # plotpath = os.path.join(plotfolder, plotname)
    # plt.savefig(plotpath)
    # print(f"Saved plot to\n {plotpath}")
    # plt.show()
    # return fig


def plot_fmae_count(
    dffp, dfall, dataset, datasplit, run_id, wofpreuse=False, logs=(10, None), x="f_mae"
):

    _df = dffp
    fpr_label = "FPreuse only" if wofpreuse else "w/o FPreuse"
    if wofpreuse:
        _df = dfall

    set_seaborn_style()

    fig, ax = plt.subplots()
    sns.histplot(
        data=_df,
        x=x,
        ax=ax,
        binwidth=0.1,
        kde=True,
        log_scale=logs,
        stat="probability",  # count percent density
        line_kws={"lw": 1},
    )
    # set xrange to 0-20
    # ax.set_xlim(0, 20)
    ax.set_xlabel(labels_human[x])
    ax.set_ylabel("Probability")
    set_style_after(ax, legend=None)

    plt.title(f"{dffp['target'].iloc[0].capitalize()} {dffp['Model'].iloc[0]} ({fpr_label})")
    plt.tight_layout()

    # save the plot
    plotname = f"{x}_count_{datasplit}-{run_id}.png"
    plotpath = os.path.join(plotfolder, plotname)
    plt.savefig(plotpath)
    print(f"Saved plot to\n {plotpath}")


def plot_step_count(dffp, dfall, dataset, datasplit, run_id, wofpreuse=False):

    _df = dffp
    fpr_label = "FPreuse only" if wofpreuse else "w/o FPreuse"
    if wofpreuse:
        _df = dfall

    set_seaborn_style()

    fig, ax = plt.subplots()
    sns.histplot(data=_df, x="nstep", ax=ax, binwidth=1, kde=True)
    # set xrange to 0-20
    ax.set_xlim(0, 20)
    ax.set_xlabel("Solver Steps")
    ax.set_ylabel("Count")
    set_style_after(ax, legend=None)

    plt.title(f"{dffp['target'].iloc[0].capitalize()} {dffp['target'].iloc[0].capitalize()} ({fpr_label})")
    plt.tight_layout()

    # save the plot
    plotname = f"step_count_{datasplit}-{run_id}.png"
    plotpath = os.path.join(plotfolder, plotname)
    plt.savefig(plotpath)
    print(f"Saved plot to\n {plotpath}")


hex_joint_kws = {
    # "gridsize": 40, # A higher value results in smaller hexbins
}


def plot_x_vs_y(
    dffp,
    dfall,
    dataset,
    datasplit,
    run_id,
    style="hist",
    ymin=None,
    ymax=None,
    logy=False,
    logx=True,
    wofpreuse=False,
    x="f_mae",
    y="nstep",
):
    set_seaborn_style()
    # x = "f_mae"
    # y = "nstep"

    label = ""

    _df = dffp
    fpr_label = "FPreuse only" if wofpreuse else "w/o FPreuse"
    if wofpreuse:
        _df = dfall
    
    try:
        _ = _df[x]
        _ = _df[y]
    except Exception as e:
        print(f'Columns in df: {_df.columns}')
        raise e


    if style == "kde":
        fig, ax = plt.subplots()
        # density plot
        sns.kdeplot(data=_df, x=x, y="nstep", ax=ax, label=label, fill=True)
    elif style == "scatter":
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=_df,
            x=x,
            y=y,
            ax=ax,
            label=label,
            # hue="z", palette="tab20",
            alpha=0.1,
        )
    elif style == "hex":
        # df.plot(kind='hexbin'
        jointgrid = sns.jointplot(
            data=_df,
            x=x,
            y=y,
            kind="hex",
            label=label,
            # cmap="crest",
            joint_kws=hex_joint_kws,
        )
        ax = jointgrid.ax_joint
    elif style == "hist":
        # df.plot(kind='hexbin'
        jointgrid = sns.jointplot(
            data=_df,
            x=x,
            y=y,
            kind="hist",
            label=label,
            # cmap="crest",
            # color="#4CB391",
        )
        ax = jointgrid.ax_joint
    else:
        raise ValueError(f"style {style} not supported")

    if ymin is not None:
        plt.ylim(bottom=ymin)
    if ymax is not None:
        plt.ylim(top=ymax)

    plt.xlabel(labels_human[x])
    plt.ylabel(labels_human[y])

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    # ax.legend()
    set_style_after(ax, legend=None)

    plt.title(f"{dffp['target'].iloc[0].capitalize()} {dffp['Model'].iloc[0]} ({fpr_label})")
    plt.tight_layout()

    # save the plot
    # plotname = f"step_vs_force_{datasplit}-{run_id}.png"
    # plotpath = os.path.join(plotfolder, plotname)
    # plt.savefig(plotpath)


def plot_step_vs_forcedelta(
    dffp,
    dfall,
    dataset,
    datasplit,
    run_id,
    metric="norm",
    style="hist",
    ymin=None,
    ymax=None,
):
    # assert metric in ["norm_mean", "norm", "cos_mean", "cos"]
    # number of solver steps vs |force_t - force_{t+1}|
    set_seaborn_style()

    # print(metric)

    x = "f_delta_" + metric

    print(f"plot_step_vs_forcedelta: plotting {len(dffp)} points.")
    if style == "kde":
        fig, ax = plt.subplots()
        # density plot
        sns.kdeplot(
            data=dffp,
            x=x,
            y="nstep",
            ax=ax,
            label="Force Delta",
            fill=True,
            # hue="z", palette="tab20"
            thresh=0.01,
            # cmap="viridis",
            # cmap="crest",
            # palette="crest",
        )
    elif style == "scatter":
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=dffp,
            x=x,
            y="nstep",
            ax=ax,
            label="Force Delta",
            # hue="z", palette="tab20",
            alpha=0.1,
        )
    elif style == "hex":
        # df.plot(kind='hexbin'
        jointgrid = sns.jointplot(
            data=dffp,
            x=x,
            y="nstep",
            kind="hex",
            label="Force Delta",
            # cmap="crest",
            color="#4CB391",
            joint_kws=hex_joint_kws,
        )
        ax = jointgrid.ax_joint
    elif style == "hist":
        # df.plot(kind='hexbin'
        jointgrid = sns.jointplot(
            data=dffp,
            x=x,
            y="nstep",
            kind="hist",
            label="Force Delta",
            # cmap="crest",
            color="#4CB391",
        )
        ax = jointgrid.ax_joint
    else:
        raise ValueError(f"style {style} not supported")

    # fit linear regression
    sns.regplot(
        data=dffp,
        x=x,
        y="nstep",
        ax=ax,
        scatter=False,
        order=2,
        line_kws=line_kws,
    )
    sns.regplot(
        data=dffp,
        x=x,
        y="nstep",
        ax=ax,
        scatter=False,
        line_kws=line_kws
        # order=2,
    )

    if metric == "norm_mean":
        xlabel = (
            r"$\frac{1}{N} \sum_{n \in atoms}^{N}|Force_t^{(n)} - Force_{t-1}^{(n)}|_2$"
        )
    elif metric == "norm":
        xlabel = r"$|Force_t^{(1\cdots n)} - Force_{t-1}^{(1\cdots n)}|_2$"
    elif metric == "cos_mean":
        xlabel = r"$\frac{1}{N} \sum_{n \in atoms}^{N} \cos(Force_t^{(n)}, Force_{t-1}^{(n)})$"
    elif metric == "cos":
        xlabel = r"$\cos(Force_t^{(1\cdots n)}, Force_{t-1}^{(1\cdots n)})$"
    elif metric == "max":
        xlabel = r"$\max_{n \in atoms}|Force_t^{(n)} - Force_{t-1}^{(n)}|_2$"
    # not delta but force at time t
    elif metric == "fnorm":
        xlabel = r"$|Force_t^{(1\cdots n)}|_2$"
    elif metric == "fnorm_max":
        xlabel = r"$max_{n \in atoms}|Force_t^{(n)}|_2$"
    elif metric == "fnorm_mean":
        xlabel = r"$\frac{1}{N} \sum_{n \in atoms}^{N}|Force_t^{(n)}|_2$"
    elif metric == "fnorm_std":
        xlabel = r"$\sigma_{n \in atoms}|Force_t^{(n)}|_2$"
    elif metric == "f_mae":
        xlabel = r"F MAE"
    else:
        raise ValueError(f"metric {metric} not supported")

    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    if ymax is not None:
        ax.set_ylim(top=ymax)

    plt.xlabel(xlabel)
    plt.ylabel("Solver Steps")
    # ax.set_yscale("log")
    # ax.legend()
    set_style_after(ax, legend=None)

    plt.title(f"{dffp['target'].iloc[0].capitalize()} (FPreuse only)")
    plt.tight_layout()

    # save the plot
    plotname = f"step_vs_forcedelta_{datasplit}-{run_id}.png"
    plotpath = os.path.join(plotfolder, plotname)
    plt.savefig(plotpath)


# pickle.dump(fig, open('FigureObject.fig.pickle', 'wb'))
# figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
# figx.show()
# data = figx.axes[0].lines[0].get_data()

# # add pos, f, e columns to the dataframe
# for i, row in df.iterrows():
#     idx = row["idx"]
#     data = collate([dataset[idx]])
#     pos = data.pos
#     f = data.y
#     e = data.e
#     df.at[i, "pos"] = pos
#     df.at[i, "f"] = f
#     df.at[i, "e"] = e
