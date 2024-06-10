import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import omegaconf
import copy
import os, sys, pathlib
import pickle

from torch_geometric.loader.dataloader import Collater


project = "EquilibriumEquiFormer"

from src.deq2ff.plotting.style import set_seaborn_style, set_style_after

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

def main(run_id, datasplit="test_fpreuse"):
    # idx_table
    table_key = f"idx_table_{datasplit}"
    # table_key = "_idx_table_{_datasplit}"

    # a = artifacts[-1]
    # table = a.get(table_key)
    # df = pd.DataFrame(table)

    api = wandb.Api()
    # https://wandb.ai/andreas-burger/EquilibriumEquiFormer/artifacts/run_table/run-0bi46nd6-idx_table_test_fpreuse/v0
    a = api.artifact(f"{project}/run-{run_id}-{table_key}:latest")
    # apath = a.download()
    table = a.get(table_key)
    df = pd.DataFrame(data=table.data, columns=table.columns)

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

    # columns: "idx", "e_mae", "f_mae", "nstep"
    # plot the loss per idx
    set_seaborn_style()

    fig, ax = plt.subplots()
    # sns.lineplot(data=df, x="idx", y="e_mae", ax=ax, label="Energy MAE")
    sns.lineplot(data=df, x="idx", y="f_mae", ax=ax, label="Force MAE")
    # ax.set_yscale("log")
    ax.set_xlabel("Index")
    ax.set_ylabel("MAE")
    ax.legend()
    set_style_after(ax)

    plt.tight_layout()

    # save the plot
    plotname = f"loss_per_idx_{datasplit}-{run_id}.png"
    plotpath = os.path.join(plotfolder, plotname)
    plt.savefig(plotpath)
    print(f"Saved plot to\n {plotpath}")

    """ 3D plot of positions, forces, energy """
    # load the data
    import equiformer.datasets.pyg.md_all as md_all

    train_dataset, val_dataset, test_dataset, test_dataset_full, all_dataset = md_all.get_md_datasets(
        root=data_path,
        dataset_arg=target,
        dname=dname,
        train_size=train_size,
        val_size=val_size,
        test_size=None, # influences data splitting
        test_size_select=test_size, # doesn't influence data splitting
        seed=seed,
        order="consecutive_all",
    )

    collate = Collater(None, None)
    # data = collate([dataset[_idx] for _idx in idx])

    # remove every patch_size row (because there is no fpreuse)
    # df = df.iloc[1:]
    test_size = run.config["test_size"]
    test_patches = run.config["test_patches"]
    patch_size = test_size // test_patches
    print(f"patch_size: {patch_size}")
    assert patch_size == 200
    # remove every patch_size row (because there is no fpreuse)
    print(f'len(df): {len(df)}')
    df = df.iloc[patch_size-1::patch_size]
    print(f'len(df): {len(df)}')

    # get idx of the minimum force mae
    # min_idx = df["f_mae"].idxmin()
    # min_idx = df["f_mae"].idxmax()
    min_idx = df["nstep"].idxmax()
    # get the data
    data = collate([test_dataset_full[min_idx]])
    pos = data.pos
    f = data.dy
    e = data.y

    side_by_side = False
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
        ax = fig.add_subplot(121, projection='3d')
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c="red", label="Atoms")
    ax.quiver(
        pos[:, 0], pos[:, 1], pos[:, 2], f[:, 0], f[:, 1], f[:, 2], 
        color="blue", label="Forces",
        **quiver_kwargs,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()

    # add previous idx to the plot
    min_idx_prev = min_idx - 1
    data = collate([test_dataset_full[min_idx_prev]])
    pos_prev = data.pos
    f_prev = data.dy
    e_prev = data.y

    if side_by_side:
        ax2 = fig.add_subplot(122, projection='3d')
    else:
        ax2 = ax
    ax2.scatter(pos_prev[:, 0], pos_prev[:, 1], pos_prev[:, 2], c="green", label="Atoms (prev)")
    ax2.quiver(
        pos_prev[:, 0], pos_prev[:, 1], pos_prev[:, 2], f_prev[:, 0], f_prev[:, 1], f_prev[:, 2], 
        color="purple", label="Forces (prev)",
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

    # pickle.dump(fig, open('FigureObject.fig.pickle', 'wb')) 
    # figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))
    # figx.show() 
    # data = figx.axes[0].lines[0].get_data()

    # # add pos, f, e columns to the dataframe
    # for i, row in df.iterrows():
    #     idx = row["idx"]
    #     data = collate([test_dataset_full[idx]])
    #     pos = data.pos
    #     f = data.y
    #     e = data.e
    #     df.at[i, "pos"] = pos
    #     df.at[i, "f"] = f
    #     df.at[i, "e"] = e






if __name__ == "__main__":

    # DEQE2FPC fpcof evalbatchsize-1 lr-5e-3 target-ethanol
    # https://wandb.ai/andreas-burger/EquilibriumEquiFormer/runs/m9lxtjgn
    # run_id = "m9lxtjgn"
    # main(run_id)

    # launchrun +use=deq target=ethanol fpreuse_test=True eval_batch_size=1 evaluate=True
    # https://wandb.ai/andreas-burger/EquilibriumEquiFormer/runs/0bi46nd6
    run_id = "0bi46nd6"
    # main(run_id)

    # launchrun +use=deq target=ethanol eval_batch_size=1 evaluate=True
    # https://wandb.ai/andreas-burger/EquilibriumEquiFormer/runs/me9p83tv
    run_id = "me9p83tv"
    # main(run_id)

    # Aspirin
    # https://wandb.ai/andreas-burger/EquilibriumEquiFormer/runs/vdx6knk5
    run_id = "vdx6knk5"
    main(run_id)