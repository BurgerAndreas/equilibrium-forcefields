import torch
from torch_geometric.data import (
    InMemoryDataset,
    download_url,
    Data,
    extract_tar,
    extract_zip,
)
import numpy as np
from torch.utils.data import Subset
import os
import os.path as osp


def convert_indices(molecule="aspirin", target_dir="datasets/md17"):

    ######################################################################################
    # Revised MD17 dataset
    # https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038
    # 100k samples

    filepath = f"datasets/rmd17/{molecule}/raw/rmd17/npz_data/rmd17_{molecule}.npz"
    raw_data = np.load(filepath)

    # https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038
    # 'old_indices' : The index of each conformation in the original MD17 dataset
    # 'old_energies' : The energy of each conformation taken from the original MD17 dataset (in units of kcal/mol)
    # 'old_forces': The forces of each conformation taken from the original MD17 dataset (in units of kcal/mol/ångstrom)
    # z = torch.from_numpy(raw_data['nuclear_charges']).long()
    # pos = torch.from_numpy(raw_data['coords']).float()
    # energy = torch.from_numpy(raw_data['energies']).float()
    # energy_old = torch.from_numpy(raw_data['old_energies']).float()
    # force_old = torch.from_numpy(raw_data['old_forces']).float()
    old_indices = torch.from_numpy(raw_data["old_indices"]).long()

    print(f"Length of revised dataset: {round(old_indices.size(0) / 1e3)}k")

    ######################################################################################
    # Original MD17 dataset
    # 211,762 samples

    filepath = f"datasets/md17/{molecule}/raw/md17_{molecule}.npz"
    raw_data_og = np.load(filepath)

    pos = torch.from_numpy(raw_data_og["R"]).float()

    print(f"Length of original dataset: {round(pos.size(0) / 1e3)}k")

    ######################################################################################
    # Conversion mapping

    original_to_revised = np.full(pos.size(0), -1)

    revised_to_original = []
    for entry in range(old_indices.size(0)):
        idx = int(old_indices[entry].item())
        revised_to_original.append(idx)
        original_to_revised[idx] = entry
    revised_to_original = np.array(revised_to_original)

    # save the mapping
    np.save(f"{target_dir}/revised_to_original_idx_{molecule}.npy", revised_to_original)
    print(f"Saved revised_to_original_idx_{molecule}.npy")

    # save the mapping
    np.save(f"{target_dir}/original_to_revised_idx_{molecule}.npy", original_to_revised)
    print(f"Saved original_to_revised_idx_{molecule}.npy")

    print("Conversion complete!\n")
    return revised_to_original, original_to_revised


def plot_revised_coverage(o2r):
    """Plot which original indices are covered by the revised dataset."""
    import matplotlib.pyplot as plt

    # if=-1, then the index is not covered
    # if>=0, then the index is covered
    # condition, True, False
    o2r = np.where(o2r >= 0, 1, 0)
    plt.scatter(x=range(len(o2r)), y=o2r, s=0.1)
    plt.xlabel("Original index")
    plt.ylabel("In revised dataset")
    plt.yticks(ticks=[0, 1], labels=["False", "True"])

    plt.title("Revised dataset coverage")
    plt.show()


if __name__ == "__main__":
    r20, o2r = convert_indices("aspirin")
    # plot_revised_coverage(o2r)
