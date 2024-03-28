import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import wandb


def main(run_id: str):
    api = wandb.Api()
    run = api.run(run_id)
    artifact = run.logged_artifacts()[0]
    table = artifact.get("fixed_point_error_traj")
    dict_table = {column: table.get_column(column) for column in table.columns}
    df = pd.DataFrame(dict_table)
    print(df)

    # plot the fixed point error trajectory
    # rel_fixed_point_error_traj on the y-axis
    # solver_step on the x-axis
    # train_step as the hue
    df = df.melt(
        id_vars=["solver_step", "train_step"], value_vars=["rel_fixed_point_error_traj"]
    )
    print(df)

    sns.lineplot(data=df, x="solver_step", y="value", hue="train_step")
    plt.show()


if __name__ == "__main__":

    # cugex52r
    run_id = "cugex52r"
    run_id = "EquilibriumEquiFormer" + "/" + run_id
    main(run_id)
