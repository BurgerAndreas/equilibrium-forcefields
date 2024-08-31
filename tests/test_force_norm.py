import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb
import os

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from deq2ff.plotting.style import chemical_symbols, plotfolder

from deq2ff.logging_utils import init_wandb
import scripts as scripts
from scripts.train_deq_md import train

# register all models
import deq2ff.register_all_models

def plot_norm_forces_per_element(args, norm_mean_atom, norm_std_atom, test_dataset_full, fname=""):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from deq2ff.plotting.style import set_seaborn_style, set_style_after
    # set everything to default
    plt.style.use('default')
    # Barplot of mean and std of norm of forces per atom type in test_dataset_full
    fig, ax = plt.subplots()
    set_seaborn_style()
    ax.bar(np.arange(len(norm_mean_atom)), norm_mean_atom)
    # ax.bar(np.arange(len(mean_atom)), mean_atom, yerr=std_atom)
    # add standard deviation
    ax.errorbar(np.arange(len(norm_mean_atom)), norm_mean_atom, yerr=norm_std_atom, fmt="none", capsize=5, c="black")
    ax.set_xticks(np.arange(len(norm_mean_atom)))
    ax.set_xticklabels(chemical_symbols[:len(norm_mean_atom)])
    ax.set_xlabel("Element")
    ax.set_ylabel("Norm of force")
    set_style_after(ax, legend=None)
    plt.title(f"{args.target.upper()}, {len(test_dataset_full)} samples")
    plt.tight_layout()
    plotname = f"norm_forces_per_element{fname}.png"
    plotpath = os.path.join(plotfolder, plotname)
    plt.savefig(plotpath)
    print(f"Saved plot to\n {plotpath}")
    plt.cla()
    plt.clf()
    plt.close()

def plot_norm_forces3d_per_element(args, mean3d_atom2, std3d_atom2, test_dataset_full, fname="", use_df=False):
    """Barplot grouped by element with three different colors for xyz."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from deq2ff.plotting.style import set_seaborn_style, set_style_after
    # set everything to default
    plt.style.use('default')
    # Barplot of mean and std of norm of forces per atom type in test_dataset_full
    set_seaborn_style()
    if use_df:
        # data to dataframe
        atom_dict = {
            "atom": chemical_symbols[:mean3d_atom2.shape[0]] * 3, 
            "mean": torch.hstack([mean3d_atom2[:, 0], mean3d_atom2[:, 1], mean3d_atom2[:, 2]]).abs(), 
            "std": torch.hstack([std3d_atom2[:, 0], std3d_atom2[:, 1], std3d_atom2[:, 2]]).abs(),
            # "mean": mean3d_atom2.flatten().abs(), 
            # "std": std3d_atom2.flatten().abs(),
            "xyz": ["x"] * len(mean3d_atom2) + ["y"] * len(mean3d_atom2) + ["z"] * len(mean3d_atom2)
        }
        _df = pd.DataFrame(atom_dict)
        # Barplot
        g = sns.catplot(
            data=_df, kind="bar",
            x="atom", y="mean", hue="xyz",
            # errorbar="sd", 
            # palette="dark", 
            # alpha=.6, 
            # height=6
        )
        # add error bars ?
        # get axes
        ax = g.axes[0,0]
        ax.set_xticks(np.arange(len(mean3d_atom2)))
        ax.set_xticklabels(chemical_symbols[:len(mean3d_atom2)])
    else:
        fig, ax = plt.subplots()
        x = np.arange(len(mean3d_atom2), dtype=float)
        xlabels = chemical_symbols[:len(mean3d_atom2)]
        width = 0.2
        offset = width / 2
        # remove rows which contain nan
        nan_mask = torch.isnan(mean3d_atom2).any(axis=1)
        mean3d_atom2 = mean3d_atom2[~nan_mask]
        std3d_atom2 = std3d_atom2[~nan_mask]
        xlabels = np.asarray(xlabels)[~nan_mask]
        x = np.arange(len(xlabels))
        # replace nan with 0
        mean3d_atom2 = mean3d_atom2.numpy()
        std3d_atom2 = std3d_atom2.numpy()
        # plot
        kws = {"width": width, "linewidth": 0, "ecolor": "gray", "capsize": 5, "error_kw": {"capsize": 5, "elinewidth": 0.5}}
        # barsabove
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.errorbar.html#matplotlib.axes.Axes.errorbar
        plt.bar(x=x-offset, height=mean3d_atom2[:, 0], label="x", yerr=std3d_atom2[:, 0], **kws)
        plt.bar(x=x, height=mean3d_atom2[:, 1], label="y", yerr=std3d_atom2[:, 1], **kws)
        plt.bar(x=x+offset, height=mean3d_atom2[:, 2], label="z", yerr=std3d_atom2[:, 2], **kws)
        
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels)

    # ax.bar(np.arange(len(mean3d_atom2)), mean3d_atom2)
    # ax.bar(np.arange(len(mean_atom)), mean_atom, yerr=std_atom)
    # add standard deviation
    # ax.errorbar(np.arange(len(mean3d_atom2)), mean3d_atom2, yerr=std3d_atom2, fmt="none", capsize=5, c="black")
    
    ax.set_xlabel("Element")
    ax.set_ylabel("|Force|")
    set_style_after(ax, loc="upper left")
    plt.title(f"{args.target.upper()}, {len(test_dataset_full)} samples")
    plt.tight_layout()
    plotname = f"norm_forces3d_per_element{fname}.png"
    plotpath = os.path.join(plotfolder, plotname)
    plt.savefig(plotpath)
    print(f"Saved plot to\n {plotpath}")
    plt.cla()
    plt.clf()
    plt.close()


@hydra.main(
    config_name="md17", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    # also load the args from equiformer/config/md17.yaml
    # and update the args with the new args (if not already present)
    # args = OmegaConf.merge(args, OmegaConf.load("equiformer/config/md17.yaml"))
    # argsmd17 = OmegaConf.load("equiformer/config/md17.yaml")
    # argsmd17.update(args)
    # args = argsmd17

    
    args.return_data = True
    args.model.max_num_elements = 10
    args.norm_forces_by_atom = "normmean"
    args.norm_forces_by_atom = "std3d"

    # init_wandb(args, project="oc20-ev2")
    run_id = init_wandb(args)

    datas = train(args)
    train_dataset = datas["train_dataset"]
    test_dataset_full = datas["test_dataset_full"]

    norm_mean_atom = torch.ones(args.model.max_num_elements)  # [A]
    norm_std_atom = torch.ones(args.model.max_num_elements)
    mean_atom = torch.ones(args.model.max_num_elements)
    std_atom = torch.ones(args.model.max_num_elements)
    mean3d_atom = torch.ones(args.model.max_num_elements, 3)
    std3d_atom = torch.ones(args.model.max_num_elements, 3)
    # concatenate all the forces
    dy = torch.cat([batch.dy for batch in train_dataset], dim=0)  # [N, 3]
    dy_norm = torch.linalg.norm(dy, dim=1)  # [N]
    atoms = torch.cat([batch.z for batch in train_dataset], dim=0)  # [N]
    # Compute statistics
    print("Computing statistics...")
    for i in tqdm(range(args.model.max_num_elements)):
        mask = atoms == i
        norm_mean_atom[i] = dy_norm[mask].mean()
        norm_std_atom[i] = dy_norm[mask].std()
        mean_atom[i] = dy[mask].mean()
        std_atom[i] = dy[mask].std()
        mean3d_atom[i] = dy[mask].mean(0)
        std3d_atom[i] = dy[mask].std(0)
    
    plot_norm_forces_per_element(args, norm_mean_atom, norm_std_atom, test_dataset_full, fname="")
    plot_norm_forces3d_per_element(args, mean3d_atom, std3d_atom, test_dataset_full, fname="")

    # norm all the forces
    print('Normalizing forces...')
    normed_full_dataset = []
    normalizer_f = datas["normalizers"]["force"]
    device = normalizer_f.device
    for i in tqdm(range(len(test_dataset_full))):
        batch = test_dataset_full[i]
        normed_full_dataset.append((
            normalizer_f.norm(
                batch.dy.to(device), batch.z.to(device)
            ),
            batch.z
        ))
    
    # recompute statistics
    norm_mean_atom2 = torch.ones(args.model.max_num_elements)
    norm_std_atom2 = torch.ones(args.model.max_num_elements)
    mean_atom2 = torch.ones(args.model.max_num_elements)
    std_atom2 = torch.ones(args.model.max_num_elements)
    mean3d_atom2 = torch.ones(args.model.max_num_elements, 3)
    std3d_atom2 = torch.ones(args.model.max_num_elements, 3)
    # concatenate all the forces
    dy = torch.cat([batch[0] for batch in normed_full_dataset], dim=0)  # [N, 3]
    dy_norm = torch.linalg.norm(dy, dim=1)  # [N]
    atoms = torch.cat([batch[1] for batch in normed_full_dataset], dim=0)  # [N]
    # Compute statistics
    for i in tqdm(range(args.model.max_num_elements)):
        mask = atoms == i
        norm_mean_atom2[i] = dy_norm[mask].mean()
        norm_std_atom2[i] = dy_norm[mask].std()
        mean_atom2[i] = dy[mask].mean()
        std_atom2[i] = dy[mask].std()
        mean3d_atom2[i] = dy[mask].mean(0)
        std3d_atom2[i] = dy[mask].std(0)
    
    plot_norm_forces_per_element(args, norm_mean_atom2, norm_std_atom2, normed_full_dataset, fname="_normed")
    plot_norm_forces3d_per_element(args, mean3d_atom2, std3d_atom2, normed_full_dataset, fname="_normed")
    
    print('Done!')


if __name__ == "__main__":
    hydra_wrapper()
