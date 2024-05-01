import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import copy
import os, sys, pathlib

"""
Fixed point convergence
abs_trace over forward-solver-iteration-steps
https://colab.research.google.com/drive/12HiUnde7qLadeZGGtt7FITnSnbUmJr-I?usp=sharing#scrollTo=V5Zff4FHqR5d
"""

entity = "andreas-burger"
project = "EquilibriumEquiFormer"

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")

# columns = ['abs', 'rel', 'solver_step', 'train_step']


def main(
    run_id: str, datasplit: str = "train", error_type="abs", ymax=None, logscale=False
):
    # https://github.com/wandb/wandb/issues/3966

    artifact_name = f"{error_type}_fixed_point_error_traj_{datasplit}"
    alias = "latest"

    api = wandb.Api()
    run = api.run(project + "/" + run_id)
    run_name = run.name

    # artifact = run.logged_artifacts()
    # print(f"len(artifact): {len(artifact)}")
    # artifact = artifact[-1]
    # table = artifact.get(artifact_name)
    # dict_table = {column: table.get_column(column) for column in table.columns}
    # df = pd.DataFrame(dict_table)

    # run = wandb.init()
    # artifact = run.use_artifact(f'{entity}/{project}/{artifact_name}:{alias}')
    # artifact_table = artifact.get(artifact_name)
    # my_df = pd.DataFrame(data=artifact_table.data, columns=artifact_table.columns)
    # wandb.finish()

    # metrics_dataframe = run.history()

    print("Downloading run history...")
    history = run.scan_history()
    print("Processing run history...")
    losses = [
        [row[artifact_name], row["_step"]]
        for row in history
        if artifact_name in row.keys()
    ]
    print(f" Losses found: {len(losses[0][0]) if len(losses) > 0 else None}")

    print(f"Filtering out None values...")
    losses_nonone = [[r, s] for r, s in losses if r is not None]
    print(f" Rows that were None: {len(losses) - len(losses_nonone)} / {len(losses)}")
    losses = losses_nonone

    print(f"Combining data into dataframe...")
    # losses = [[r, s, [*range(len(r))]] for r, s in losses]
    losses = [
        {
            error_type: pd.Series(r),
            "train_step": pd.Series([s] * len(r)),
            "solver_step": pd.Series(range(len(r))),
        }
        for r, s in losses
    ]
    losses_concat = {
        k: pd.concat([d[k] for d in losses], axis=0) for k in losses[0].keys()
    }
    # print(f"losses_concat: {losses_concat}")
    df = pd.DataFrame(losses_concat)

    # dataframe with three colums: error_type, train_step, solver_step
    # print(df)

    # plot: x=solver_step, y=error_type, hue=train_step
    sns.lineplot(data=df, x="solver_step", y=error_type, hue="train_step")
    plt.xlabel("Fixed-point solver step")
    plt.ylabel(f"Fixed-point error ({error_type})")
    if logscale:
        plt.yscale("log")
    if ymax is not None:
        # cant plot 0 on logscale
        # plt.ylim(1e-12, ymax)
        plt.ylim(top=ymax)
    # legend title
    plt.title(f"{run_name}")

    fname = f"{plotfolder}/fixed_point_error_traj_{datasplit}_{run_id.split('/')[-1]}_{error_type}.png"
    plt.savefig(fname)
    print(f"Saved plot to {fname}")

    # close the plot
    plt.close()
    plt.gca().clear()
    plt.gcf().clear()


if __name__ == "__main__":

    # ----------------- E2 -----------------
    # E2 aauf 8uuq632s
    # not converged
    run_id = "8uuq632s"
    # main(run_id, error_type="abs", datasplit="train", logscale=True)

    # E2 normlayer-norm aauf 8dqpu458
    # not converged
    # run_id = "8dqpu458"
    # main(run_id, error_type="abs", datasplit="train", logscale=True)

    # E2 fmaxiter-10 aauf c7jgk0p7
    # not converged
    run_id = "c7jgk0p7"
    # main(run_id, error_type="abs", datasplit="train", logscale=True)

    # E2 anderson add-inj-prev (aa) 1hwg7al5
    # not converged
    run_id = "1hwg7al5"
    # main(run_id, error_type="abs", datasplit="train", logscale=True)

    # E2 anderson by8radv7
    # somewhat converged
    run_id = "by8radv7"
    # main(run_id, error_type="abs", datasplit="train", logscale=True)

    # E2 nl7jlh8q
    # somewhat converged
    run_id = "nl7jlh8q"
    # main(run_id, error_type="abs", datasplit="train", logscale=True)

    # E2 fsolver-broyden alphadrop-0 droppathrate-0 1hjry1oh
    run_id = "1hjry1oh"
    main(run_id, error_type="abs", datasplit="train", logscale=True)

    # E2 alphadrop-0 droppathrate-0 12uk3wdo
    run_id = "12uk3wdo"
    main(run_id, error_type="abs", datasplit="train", logscale=True)

    # ----------------- E1 -----------------
    # broyden pathnorm: f9bg18sp
    # main("f9bg18sp", datasplit="train")
    # main("f9bg18sp", datasplit="train", ymax=0.01)
    # main("f9bg18sp", datasplit="train", logscale=True)
    # main("f9bg18sp", datasplit="train", logscale=True, ymax=0.001)
    # main("f9bg18sp", error_type='rel', datasplit="train", logscale=True, ymax=0.001)
    # main("f9bg18sp", error_type='rel', datasplit="train", logscale=True, ymax=0.001)

    # broyden pathnorm 64precision
    # run_id = "6cfnokgr"
    # main(run_id, error_type="abs", datasplit="train", logscale=True)
    # main(run_id, error_type="rel", datasplit="train", logscale=True)
    # main(run_id, error_type='abs64', datasplit="train", logscale=True)
    # main(run_id, error_type='rel64', datasplit="train", logscale=True)
    # main(run_id, error_type='abs64', datasplit="train", logscale=True, ymax=0.001)
    # main(run_id, error_type='rel64', datasplit="train", logscale=True, ymax=0.001)

    # broyden: iptk3b73
    # anderson: neo7e1vi
    # z0-ones: gzifpvwe
    # 6 layers: yuqbla4u
    # FPreuse: auffq4x0
    # Tanh: ii3gls8d
