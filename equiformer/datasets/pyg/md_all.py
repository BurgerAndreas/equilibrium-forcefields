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

import wandb

# from typing import Callable, List, Optional, Union


np.int = np.int32
np.float = np.float64
np.bool = np.bool_


class MDAll(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the revised versions.
    https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038

    If you are using a newer version of torch_geometric, just use their implementation:
    https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MD17.html

    Usage:
        train_dataset, val_dataset, test_dataset = md17_dataset.get_md_datasets(
            root=os.path.join(args.data_path, 'md17', args.target),
            dataset_arg=args.target,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=None,
            seed=args.seed,
        )
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/"  # gdml
    revised_url = "https://archive.materialscloud.org/record/file?filename=rmd17.tar.bz2&record_id=466"


    # We note that the file names have been changed.
    # For example, `aspirin_dft` -> `md17_aspirin`
    # See https://github.com/pyg-team/pytorch_geometric/commit/213f0ff95140eb1a1fbf7d99b012d458ef360f71#diff-a85570faabaf1806684e5b6654deed3863273bbe703f237846accd11948f4675
    # https://github.com/pyg-team/pytorch_geometric/pull/6734
    molecule_files = {
        "md17": {
            'aspirin': "md17_aspirin.npz",
            'benzene': "md17_benzene2017.npz",
            'ethanol': "md17_ethanol.npz",
            'malonaldehyde': "md17_malonaldehyde.npz",
            'naphthalene': "md17_naphthalene.npz",
            'salicylic_acid': "md17_salicylic.npz",
            'toluene': "md17_toluene.npz",
            'uracil': "md17_uracil.npz",
        },
        'extended': {
            "paracetamol": "paracetamol_dft.npz",
            "azobenzene": "azobenzene_dft.npz",
        },
        'revised': {
            "revised benzene": "rmd17_benzene.npz",
            "revised uracil": "rmd17_uracil.npz",
            "revised naphthalene": "rmd17_naphthalene.npz",
            "revised aspirin": "rmd17_aspirin.npz",
            "revised salicylic acid": "rmd17_salicylic.npz",
            "revised malonaldehyde": "rmd17_malonaldehyde.npz",
            "revised ethanol": "rmd17_ethanol.npz",
            "revised toluene": "rmd17_toluene.npz",
            "revised paracetamol": "rmd17_paracetamol.npz",
            "revised azobenzene": "rmd17_azobenzene.npz",
        },
        'ccsd': {
            "benzene CCSD(T)": "benzene_ccsd_t.zip",
            "aspirin CCSD": "aspirin_ccsd.zip",
            "malonaldehyde CCSD(T)": "malonaldehyde_ccsd_t.zip",
            "ethanol CCSD(T)": "ethanol_ccsd_t.zip",
            "toluene CCSD(T)": "toluene_ccsd_t.zip",
            "benzene FHI-aims": "benzene2018_dft.npz",
        },
        'md22': {
            "AT-AT-CG-CG": "md22_AT-AT-CG-CG.npz",
            "AT-AT": "md22_AT-AT.npz",	
            "Ac-Ala3-NHMe": "md22_Ac-Ala3-NHMe.npz",	
            "DHA": "md22_DHA.npz",
            "buckyball-catcher": "md22_buckyball-catcher.npz",
            "dw-nanotube": "md22_dw_nanotube.npz",
            "stachyose": "md22_stachyose.npz",
        },
    }
    molecule_files['rmd17og'] = molecule_files['md17']

    def __init__(
        self,
        root,
        dataset_arg,
        transform=None,
        pre_transform=None,
        dname='md17',
        # revised=False,
        # revised_old=False,
        # ccsd=False,
    ):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'."
        )
        assert dname in MDAll.molecule_files.keys(), f"Unknown data arguments {dname}. Try: {MDAll.molecule_files.keys()}"
        if dname == 'rmd17':
            assert 'revised ' + dataset_arg in MDAll.molecule_files[dname], f"Unknown data arguments {dataset_arg} with {dname}. Try: {MDAll.available_molecules[dname]}"
        else:
            assert dataset_arg in MDAll.molecule_files[dname], f"Unknown data arguments {dataset_arg} with {dname}. Try: {MDAll.available_molecules[dname]}"

        self.name = dataset_arg
        self.dname = dname

        # For simplicity, always use one type of molecules
        """
        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.molecule_files)
        """
        self.molecules = dataset_arg.split(",")

        """
        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )
        """

        super(MDAll, self).__init__(root, transform, pre_transform)

        self.offsets = [0]
        self.data_all, self.slices_all = [], []
        for path in self.processed_paths:
            data, slices = torch.load(path)
            self.data_all.append(data)
            self.slices_all.append(slices)
            self.offsets.append(
                len(slices[list(slices.keys())[0]]) - 1 + self.offsets[-1]
            )

    def len(self):
        return sum(
            len(slices[list(slices.keys())[0]]) - 1 for slices in self.slices_all
        )

    def get(self, idx):
        data_idx = 0
        while data_idx < len(self.data_all) - 1 and idx >= self.offsets[data_idx + 1]:
            data_idx += 1
        self.data = self.data_all[data_idx]
        self.slices = self.slices_all[data_idx]
        return super(MDAll, self).get(idx - self.offsets[data_idx])

    # Dataset
    # @property
    # def raw_paths(self) -> List[str]:
    #     r"""The absolute filepaths that must be present in order to skip
    #     downloading."""
    #     files = to_list(self.raw_file_names)
    #     return [osp.join(self.raw_dir, f) for f in files]

    # @property
    # def processed_paths(self) -> List[str]:
    #     r"""The absolute filepaths that must be present in order to skip
    #     processing."""
    #     files = to_list(self.processed_file_names)
    #     return [osp.join(self.processed_dir, f) for f in files]

    # MD17
    @property
    def raw_file_names(self):
        if self.dname == 'rmd17':
            return [
                osp.join(
                    "rmd17", "npz_data", MDAll.molecule_files[self.dname][f"revised {mol}"]
                )
                for mol in self.molecules
            ]
        else:
            return [MDAll.molecule_files[self.dname][mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        if self.dname == 'rmd17og':
            return [f"rmd17og-{mol}.pt" for mol in self.molecules]
        elif self.dname == 'rmd17':
            return [f"rmd17-{mol}.pt" for mol in self.molecules]
        elif self.dname == 'md22':
            return [f"md22-{mol}.pt" for mol in self.molecules]
        elif self.dname == 'md17':
            return [f"md17-{mol}.pt" for mol in self.molecules]
        elif self.dname == 'ccsd':
            return [f"ccsd-{mol}.pt" for mol in self.molecules]
        else:
            raise ValueError(f"Unknown dataset name: {self.dname}")

    # PyG MD17
    # @classmethod
    # def save(cls, data_list: Sequence[BaseData], path: str) -> None:
    #     r"""Saves a list of data objects to the file path :obj:`path`."""
    #     data, slices = cls.collate(data_list)
    #     fs.torch_save((data.to_dict(), slices, data.__class__), path)

    def download(self):
        for file_name in self.raw_file_names:
            if self.dname == 'rmd17':
                path = download_url(self.revised_url, self.raw_dir)
                extract_tar(path, self.raw_dir, mode="r:bz2")
                os.unlink(path)
            else:
                # download_url(MD17.raw_url + file_name, self.raw_dir)
                url = f"{self.raw_url}/{file_name}"
                path = download_url(url, self.raw_dir)
                if self.dname == 'ccsd':
                    extract_zip(path, self.raw_dir)
                    os.unlink(path)

    def process(self):
        # rm -r -f datasets/md17/**/processed

        print(
            "Saving processed data to",
            self.processed_dir,
            f"processed_file_names={self.processed_file_names}",
        )
        print(f"self.pre_filter: {self.pre_filter}")
        print(f"self.pre_transform: {self.pre_transform}")

        it = zip(self.raw_paths, self.processed_paths)
        old_indices = None
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)
            print(f'keys in raw_data: {raw_data.keys()}')
            if self.dname == 'rmd17og':
                z = torch.from_numpy(raw_data["nuclear_charges"]).long()
                pos = torch.from_numpy(raw_data["coords"]).float()
                # https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038
                # 'old_indices' : The index of each conformation in the original MD17 dataset
                # 'old_energies' : The energy of each conformation taken from the original MD17 dataset (in units of kcal/mol)
                # 'old_forces': The forces of each conformation taken from the original MD17 dataset (in units of kcal/mol/ångstrom)
                energy = torch.from_numpy(raw_data["old_energies"]).float()
                force = torch.from_numpy(raw_data["old_forces"]).float()
                old_indices = torch.from_numpy(raw_data["old_indices"]).long()
            elif self.dname == 'rmd17':
                z = torch.from_numpy(raw_data["nuclear_charges"]).long()
                pos = torch.from_numpy(raw_data["coords"]).float()
                energy = torch.from_numpy(raw_data["energies"]).float()  # [100000]
                force = torch.from_numpy(raw_data["forces"]).float()  # [100000, 21, 3]
            else:
                # md17, md22, ccsd
                z = torch.from_numpy(raw_data["z"]).long()
                pos = torch.from_numpy(raw_data["R"]).float()
                energy = torch.from_numpy(raw_data["E"]).float()  # [211762, 1]
                force = torch.from_numpy(raw_data["F"]).float()

            data_list = []
            print("Dataset size:", pos.size(0))
            for i in range(pos.size(0)):
                # old: ['z', 'pos', 'batch', 'y', 'dy']
                # new: ['z', 'pos', 'energy', 'force']
                e = energy[i]
                if e.shape == torch.Size([]):
                    e = e.unsqueeze(0)
                if e.shape == torch.Size([1]):
                    e = e.unsqueeze(1)
                # f: [21, 3]
                assert e.shape == torch.Size([1, 1]), f"Energy shape: {e.shape}"
                # put together
                data = Data(
                    z=z,
                    pos=pos[i],
                    y=e,
                    dy=force[i],
                    idx=i,
                    old_idx=None if old_indices is None else old_indices[i],
                )
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            # self.save(data_list, processed_path)
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)


def fix_train_val_test_size(train_size, val_size, test_size, dset_len):
    assert (train_size is None) + (val_size is None) + (
        test_size is None
    ) <= 1, "Only one of train_size, val_size, test_size is allowed to be None."
    is_float = (
        isinstance(train_size, float),
        isinstance(val_size, float),
        isinstance(test_size, float),
    )

    # if we provide the sizes as percentages
    train_size = round(dset_len * train_size) if is_float[0] else train_size
    val_size = round(dset_len * val_size) if is_float[1] else val_size
    test_size = round(dset_len * test_size) if is_float[2] else test_size

    if train_size is None:
        train_size = dset_len - val_size - test_size
    elif val_size is None:
        val_size = dset_len - train_size - test_size
    elif test_size is None:
        test_size = dset_len - train_size - val_size

    if train_size + val_size + test_size > dset_len:
        if is_float[2]:
            test_size -= 1
        elif is_float[1]:
            val_size -= 1
        elif is_float[0]:
            train_size -= 1

    assert train_size >= 0 and val_size >= 0 and test_size >= 0, (
        f"One of training ({train_size}), validation ({val_size}) or "
        f"testing ({test_size}) splits ended up with a negative size."
    )
    return train_size, val_size, test_size


# From https://github.com/torchmd/torchmd-net/blob/72cdc6f077b2b880540126085c3ed59ba1b6d7e0/torchmdnet/utils.py#L54
def train_val_test_split(dset_len, train_size, val_size, test_size, seed, order=None):

    train_size, val_size, test_size = fix_train_val_test_size(
        train_size, val_size, test_size, dset_len
    )

    total = train_size + val_size + test_size

    assert dset_len >= total, (
        f"The dataset ({dset_len}) is smaller than the "
        f"combined split sizes ({total} = {train_size} + {val_size} + {test_size})."
    )
    if total < dset_len:
        print(f"{dset_len - total} samples were excluded from the dataset")

    idxs = np.arange(dset_len, dtype=np.int)
    # shuffle the indices
    if order is None:
        idxs = np.random.default_rng(seed).permutation(idxs)

    idx_train = idxs[:train_size]
    idx_val = idxs[train_size : train_size + val_size]
    idx_test = idxs[train_size + val_size : total]

    if order is None:
        return np.array(idx_train), np.array(idx_val), np.array(idx_test)

    # ids are sampled consecutively -> important for fixed-point reuse
    elif order == "consecutive":
        print("Dataset: Using consecutive order")
        return idx_train, idx_val, idx_test

    elif order == "consecutive_test":
        print("Dataset: Using consecutive order for test set")
        # V1: train and val are shuffled, test is consecutive
        # less diverse train and val, but test is guaranteed to be different
        # idxs = idxs[: train_size + val_size]
        # idxs = np.random.default_rng(seed).permutation(idxs)
        # idx_train = idxs[:train_size]
        # idx_val = idxs[train_size:]
        # V2: same-as-default train and val, but test might overlap with train
        idxs_rand = np.random.default_rng(seed).permutation(idxs)
        idx_train = idxs_rand[:train_size]
        idx_val = idxs_rand[train_size : train_size + val_size]
        # test: random consecutive block
        test_start_idx = np.random.randint(0, dset_len - test_size -1)
        idx_test = idxs[test_start_idx : test_start_idx + test_size]
        return np.array(idx_train), np.array(idx_val), idx_test

    else:
        idx_train = [order[i] for i in idx_train]
        idx_val = [order[i] for i in idx_val]
        idx_test = [order[i] for i in idx_test]
        return np.array(idx_train), np.array(idx_val), np.array(idx_test)


# From: https://github.com/torchmd/torchmd-net/blob/72cdc6f077b2b880540126085c3ed59ba1b6d7e0/torchmdnet/utils.py#L112
def make_splits(
    dataset_len,
    train_size,
    val_size,
    test_size,
    seed,
    # max_samples=-1, # take the first max_samples samples from the dataset
    filename=None,  # path to save split index
    splits=None,  # load split
    order=None,
):
    """
    splits: path to a .npz file containing the splits or a dict of paths to .npz files containing the splits. Ignored if order is not None.
    order: order of the dataset, e.g. consecutive, consecutive_test, or a list of indices.
    """
    if splits is None or order is not None:
        idx_train, idx_val, idx_test = train_val_test_split(
            dataset_len, train_size, val_size, test_size, seed, order
        )
        # save the randomly created split
        if order is None:
            if filename is not None:
                np.savez(
                    filename, idx_train=idx_train, idx_val=idx_val, idx_test=idx_test
                )

    elif type(splits) == dict:
        train_size, val_size, test_size = fix_train_val_test_size(
            train_size, val_size, test_size, dataset_len
        )
        # datasets/rmd17/aspirin/raw/rmd17/splits/index_test_01.csv
        # dtype = np.dtype('int64')
        dtype = np.dtype("int32")
        idx_train = np.loadtxt(splits["train"], dtype=dtype)
        idx_train = idx_train[:train_size]
        # test and val are combined
        idx_test_val = np.loadtxt(splits["test"], dtype=dtype)
        idx_val = idx_test_val[:val_size]
        idx_test = idx_test_val[val_size : val_size + test_size]
        print(f"Loaded splits from {splits['train']} and {splits['test']}")

    else:
        splits = np.load(splits)
        idx_train = splits["idx_train"]
        idx_val = splits["idx_val"]
        idx_test = splits["idx_test"]

    return (
        torch.from_numpy(idx_train),
        torch.from_numpy(idx_val),
        torch.from_numpy(idx_test),
    )


def get_md_datasets(
    root,
    dataset_arg,
    train_size,
    val_size,
    test_size,
    seed,
    # added
    dname='md17',
    max_samples: int = -1,
    order=None,
    return_idx=False,
    load_splits=None,
):
    """
    Return training, validation and testing sets of MD17 with the same data partition as TorchMD-NET.

    Args:
        rmd17: use revised version of MD17 with more accurate energies and forces 
        rmd17og: Use the non-revised (old) data downloaded from the revised dataset and processed with the revised dataset's preprocessing script (bool)
        max_samples: take the first max_samples samples from the dataset. Ensures reproducibility between datasets of different lengths.

    """

    # root: "datasets/md17/aspirin"
    root = os.path.join(root, dname, dataset_arg)
    if dname == "md17og":
        root = root.replace("md17og", "md17")

    # keys: ['z', 'pos', 'batch', 'y', 'dy']
    all_dataset = MDAll(root, dataset_arg, dname=dname)

    if load_splits == False:
        load_splits = None
    if load_splits == True and (dname in ["md17", "rmd17"]):
        order = None
        # datasets/rmd17/aspirin/raw/rmd17/splits/index_test_01.csv
        load_splits = {
            "train": osp.join(root, "raw", "rmd17", "splits", "index_train_01.csv"),
            "test": osp.join(root, "raw", "rmd17", "splits", "index_test_01.csv"),
        }

    # different dataset lengths will lead to different permutations
    # hard setting the max_samples to ensure reproducibility
    dset_len = len(all_dataset)
    if max_samples > 0:
        _total = train_size if isinstance(train_size, int) else 0
        _total += val_size if isinstance(val_size, int) else 0
        _total += test_size if isinstance(test_size, int) else 0
        if _total > 0:
            assert (
                max_samples >= _total
            ), f"max_samples ({max_samples}) must be greater than the sum of train_size, val_size, and test_size ({_total})"
        print(
            f"Taking the first {max_samples} samples from the dataset (length {dset_len})."
        )
        dset_len = min(int(max_samples), dset_len)

    idx_train, idx_val, idx_test = make_splits(
        dset_len,
        train_size,
        val_size,
        test_size,
        seed,
        # save splits
        filename=os.path.join(root, "splits.npz"),
        # load splits
        splits=load_splits,
        # idx are consecutive -> important for fixed-point reuse
        order=order,
    )

    if return_idx:
        return idx_train, idx_val, idx_test
    
    # log split to wandb
    if wandb.run is not None:
        max_num = 1000
        # wandb.log({"idx_train": idx_train[:max_num].tolist(), "idx_val": idx_val[:max_num].tolist(), "idx_test": idx_test[:max_num].tolist()}, step=0)
        wandb.log({"idx_train": idx_train[:max_num], "idx_val": idx_val[:max_num], "idx_test": idx_test[:max_num]}, step=0)

    train_dataset = Subset(all_dataset, idx_train)
    val_dataset = Subset(all_dataset, idx_val)
    test_dataset = Subset(all_dataset, idx_test)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":

    from torch_geometric.loader import DataLoader

    _root_path = "./datasets/test"
    train_dataset, val_dataset, test_dataset = get_md_datasets(
        root=_root_path,
        dataset_arg="aspirin",
        train_size=950,
        val_size=50,
        test_size=None,
        seed=1,
        revised=True,
    )

    print("Training set size:   {}".format(len(train_dataset)))
    print("Validation set size: {}".format(len(val_dataset)))
    print("Testing set size:    {}".format(len(test_dataset)))

    print("train_dataset[2]", train_dataset[2])

    train_loader = DataLoader(train_dataset, batch_size=8)
    for i, data in enumerate(train_loader):
        print("data", data)
        print("data.y", data.y)
        break
