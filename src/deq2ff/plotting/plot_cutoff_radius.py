import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from torch_cluster import radius_graph
from torch_scatter import scatter

from e3nn import o3

import os
import pathlib
import sys
from typing import Iterable, Optional

import equiformer.datasets.pyg.md_all as rmd17_dataset

from equiformer.logger import FileLogger

# import equiformer.nets
from equiformer.nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from equiformer.optim_factory import create_optimizer

from equiformer.engine import AverageMeter, compute_stats

import hydra
import wandb
import omegaconf
from omegaconf import DictConfig

import inspect

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

"""
Test how many edges between atoms (nodes) we get as a function of the cutoff distance max_radius.
As equivariant tensor products are the computational bottleneck, we want to minimize the number of edges.
"""

# parent folder of the plot
plotfolder = pathlib.Path(__file__).parent.absolute()
plotfolder = os.path.join(plotfolder, "plots")


def test(args, max_radius=np.arange(1.0, 10.0), batch_size=1):

    args.batch_size = batch_size

    """ Dataset """
    if args.use_original_datasetcreation:
        import equiformer.datasets.pyg.md17 as md17_dataset

        train_dataset, val_dataset, test_dataset = md17_dataset.get_md17_datasets(
            root=os.path.join(args.data_path, "md17", args.target),
            dataset_arg=args.target,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=None,
            seed=args.seed,
            # order="consecutive_test" if args.fpreuse_test else None,
        )
    else:
        import equiformer.datasets.pyg.md_all as md_all

        train_dataset, val_dataset, test_dataset = md_all.get_md_datasets(
            root=args.data_path,
            dataset_arg=args.target,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=None,
            seed=args.seed,
            dname=args.dname,
            order=md_all.get_order,
        )

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Data Loader """
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    """ Network """
    create_model = model_entrypoint(args.model.name)
    if "deq_kwargs" in args:
        model = create_model(
            task_mean=mean, task_std=std, **args.model, **args.deq_kwargs
        )
    else:
        model = create_model(task_mean=mean, task_std=std, **args.model)

    model = model.to(device)
    model.train()

    num_edges = []
    for i, max_radius in enumerate(max_radius):
        args.model.max_radius = max_radius.item()
        model.max_radius = max_radius.item()

        for step, data in enumerate(train_loader):
            data = data.to(device)

            node_atom = data.z
            pos = data.pos
            batch = data.batch

            if step == 0 and i == 0:
                n = node_atom.shape[0]
                print("")
                print(f"batch_size: {args.batch_size}")
                print(f"number of atoms: {n}")
                print(f"max number of directed edges = n(n-1) = {n * (n - 1)}")
                print("")

            # atom type z_i
            atom_embedding, atom_attr, atom_onehot = model.atom_embed(node_atom)

            # get graph edges based on radius
            edge_src, edge_dst = radius_graph(
                x=pos, r=model.max_radius, batch=batch, max_num_neighbors=1000
            )
            edge_vec = pos.index_select(0, edge_src) - pos.index_select(0, edge_dst)
            # radial basis function embedding of edge length
            edge_length = edge_vec.norm(dim=1)
            edge_length_embedding = model.rbf(edge_length)
            # spherical harmonics embedding of edge vector
            edge_sh = o3.spherical_harmonics(
                l=model.irreps_edge_attr,
                x=edge_vec,
                normalize=True,
                normalization="component",
            )
            # Constant One, r_ij -> Linear, Depthwise TP, Linear, Scaled Scatter
            edge_degree_embedding = model.edge_deg_embed(
                # atom_embedding is just used for the shape
                atom_embedding,
                edge_sh,
                edge_length_embedding,
                edge_src,
                edge_dst,
                batch,
            )

            # node_features = x
            node_features = atom_embedding + edge_degree_embedding

            # node_attr = ?
            # node_attr = torch.ones_like(node_features.narrow(dim=1, start=0, length=1))
            node_attr = torch.ones_like(node_features.narrow(1, 0, 1))

            print(f"\nmax_radius: {max_radius}")
            print(f"node_attr: {node_attr.shape}")
            print(f"edge_src: {edge_src.shape}")
            print(f"edge_dst: {edge_dst.shape}")
            print(f"edge_sh: {edge_sh.shape}")

            ne = edge_src.shape[0]
            num_edges.append(ne)
            break

    return num_edges


def plot_num_edges_over_max_radius(max_radius, num_edges):
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    plt.plot(max_radius, num_edges, marker="o")
    plt.xlabel("Max radius for edges [Angstrom]")
    plt.ylabel("Number of edges per atom")
    # vertical line at 5.0 Angstrom
    plt.axvline(x=5.0, color="gray", linestyle="--")
    plt.title("Number of directed edges as a function of max_radius")
    plt.savefig(f"{plotfolder}/num_edges_over_max_radius.png")
    print(f"Saved plot to {plotfolder}/num_edges_over_max_radius.png")


@hydra.main(
    config_name="md17",
    config_path="../../../equiformer/config",
    version_base="1.3",
)
def hydra_wrapper(args: DictConfig) -> None:

    # for deq:
    # args.model.name = "deq_graph_attention_transformer_l2_md17"

    # run tests
    max_radius = np.arange(1.0, 10.0)
    num_edges = test(args, max_radius=max_radius, batch_size=1)
    plot_num_edges_over_max_radius(max_radius, num_edges)

    print("\n")
    print("Done!")


if __name__ == "__main__":
    hydra_wrapper()
