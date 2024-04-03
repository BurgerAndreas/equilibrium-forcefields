import torch
from torch_geometric.data import InMemoryDataset, download_url, Data, extract_tar, extract_zip
import numpy as np
from torch.utils.data import Subset
import os
import os.path as osp

from typing import Callable, List, Optional, Union


np.int = np.int32
np.float = np.float64
np.bool = np.bool_


class MD17(InMemoryDataset):
    """Machine learning of accurate energy-conserving molecular force fields (Chmiela et al. 2017)
    This class provides functionality for loading MD trajectories from the original dataset, not the revised versions.
    See http://www.quantum-machine.org/gdml/#datasets for details.

    Usage:
        train_dataset, val_dataset, test_dataset = md17_dataset.get_md17_datasets(
            root=os.path.join(args.data_path, args.target),
            dataset_arg=args.target,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=None,
            seed=args.seed,
        )
    """

    raw_url = "http://www.quantum-machine.org/gdml/data/npz/" # gdml
    revised_url = ('https://archive.materialscloud.org/record/'
                   'file?filename=rmd17.tar.bz2&record_id=466')

    # We note that the file names have been changed.
    # For example, `aspirin_dft` -> `md17_aspirin`
    # See https://github.com/pyg-team/pytorch_geometric/commit/213f0ff95140eb1a1fbf7d99b012d458ef360f71#diff-a85570faabaf1806684e5b6654deed3863273bbe703f237846accd11948f4675
    # https://github.com/pyg-team/pytorch_geometric/pull/6734
    molecule_files = dict(
        aspirin="md17_aspirin.npz",
        benzene="md17_benzene2017.npz",
        ethanol="md17_ethanol.npz",
        malonaldehyde="md17_malonaldehyde.npz",
        naphthalene="md17_naphthalene.npz",
        salicylic_acid="md17_salicylic.npz",
        toluene="md17_toluene.npz",
        uracil="md17_uracil.npz",
    )

    molecule_files_extended = {
        'paracetamol': 'paracetamol_dft.npz',
        'azobenzene': 'azobenzene_dft.npz',
    }

    molecule_files_revised = {
        'revised benzene': 'rmd17_benzene.npz',
        'revised uracil': 'rmd17_uracil.npz',
        'revised naphthalene': 'rmd17_naphthalene.npz',
        'revised aspirin': 'rmd17_aspirin.npz',
        'revised salicylic acid': 'rmd17_salicylic.npz',
        'revised malonaldehyde': 'rmd17_malonaldehyde.npz',
        'revised ethanol': 'rmd17_ethanol.npz',
        'revised toluene': 'rmd17_toluene.npz',
        'revised paracetamol': 'rmd17_paracetamol.npz',
        'revised azobenzene': 'rmd17_azobenzene.npz',
    }

    molecules_files_CCSD = {
        'benzene CCSD(T)': 'benzene_ccsd_t.zip',
        'aspirin CCSD': 'aspirin_ccsd.zip',
        'malonaldehyde CCSD(T)': 'malonaldehyde_ccsd_t.zip',
        'ethanol CCSD(T)': 'ethanol_ccsd_t.zip',
        'toluene CCSD(T)': 'toluene_ccsd_t.zip',
        'benzene FHI-aims': 'benzene2018_dft.npz',
    }

    available_molecules = list(molecule_files.keys())

    # revised dataset
    # https://archive.materialscloud.org/record/file?record_id=466&filename=rmd17.tar.bz2
    # All the revised trajectories are available by changing the name from e.g. benzene to revised benzene

    def __init__(self, root, dataset_arg, transform=None, pre_transform=None, revised=False, ccsd=False):
        assert dataset_arg is not None, (
            "Please provide the desired comma separated molecule(s) through"
            f"'dataset_arg'. Available molecules are {', '.join(MD17.available_molecules)} "
            "or 'all' to train on the combined dataset."
        )
        assert dataset_arg in MD17.available_molecules, "Unknown data argument"

        if revised:
            root = root.replace('md17', 'rmd17').replace('rrmd17', 'rmd17')
        else:
            print(f'\nWarning: Using the original MD17 dataset. Please consider using the revised version (equiformer/datasets/pyg/md17.py).\n')

        self.name = dataset_arg
        self.revised = revised
        self.ccsd = ccsd

        # For simplicity, always use one type of molecules
        """
        if dataset_arg == "all":
            dataset_arg = ",".join(MD17.available_molecules)
        """
        self.molecules = dataset_arg.split(",")

        """
        if len(self.molecules) > 1:
            rank_zero_warn(
                "MD17 molecules have different reference energies, "
                "which is not accounted for during training."
            )
        """

        super(MD17, self).__init__(root, transform, pre_transform)

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
        return super(MD17, self).get(idx - self.offsets[data_idx])

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
        if self.revised:
            return [osp.join('rmd17', 'npz_data', MD17.molecule_files_revised[f'revised {mol}']) for mol in self.molecules]
        else:
            return [MD17.molecule_files[mol] for mol in self.molecules]

    @property
    def processed_file_names(self):
        if self.revised:
            return [f"rmd17-{mol}.pt" for mol in self.molecules]
        else:
            return [f"md17-{mol}.pt" for mol in self.molecules]

    # PyG MD17
    # @property
    # def raw_dir(self) -> str:
    #     if self.revised:
    #         return osp.join(self.root, 'raw')
    #     return osp.join(self.root, self.name, 'raw')

    # @property
    # def processed_dir(self) -> str:
    #     return osp.join(self.root, self.name, 'processed')

    # @property
    # def raw_file_names(self) -> Union[str, List[str]]:
    #     name = self.file_names[self.name]
    #     if self.revised:
    #         return osp.join('rmd17', 'npz_data', name)
    #     elif self.ccsd:
    #         return [name[:-4] + '-train.npz', name[:-4] + '-test.npz']
    #     return name
    
    # @property
    # def processed_file_names(self) -> List[str]:
    #     if self.ccsd:
    #         return ['train.pt', 'test.pt']
    #     else:
    #         return ['data.pt']
    
    # @classmethod
    # def save(cls, data_list: Sequence[BaseData], path: str) -> None:
    #     r"""Saves a list of data objects to the file path :obj:`path`."""
    #     data, slices = cls.collate(data_list)
    #     fs.torch_save((data.to_dict(), slices, data.__class__), path)

    def download(self):
        for file_name in self.raw_file_names:
            if self.revised:
                path = download_url(self.revised_url, self.raw_dir)
                extract_tar(path, self.raw_dir, mode='r:bz2')
                os.unlink(path)
            else:
                # download_url(MD17.raw_url + file_name, self.raw_dir)
                url = f'{self.raw_url}/{file_name}'
                path = download_url(url, self.raw_dir)
                if self.ccsd:
                    extract_zip(path, self.raw_dir)
                    os.unlink(path)

    def process(self):

        it = zip(self.raw_paths, self.processed_paths)
        for raw_path, processed_path in it:
            raw_data = np.load(raw_path)
            if self.revised:
                z = torch.from_numpy(raw_data['nuclear_charges']).long()
                pos = torch.from_numpy(raw_data['coords']).float()
                energy = torch.from_numpy(raw_data['energies']).float()
                force = torch.from_numpy(raw_data['forces']).float()
            else:
                z = torch.from_numpy(raw_data['z']).long()
                pos = torch.from_numpy(raw_data['R']).float()
                energy = torch.from_numpy(raw_data['E']).float()
                force = torch.from_numpy(raw_data['F']).float()

            data_list = []
            print('dataset size:', pos.size(0))
            for i in range(pos.size(0)):
                # old: ['z', 'pos', 'batch', 'y', 'dy']
                # new: ['z', 'pos', 'energy', 'force']
                data = Data(z=z, pos=pos[i], y=energy[i], dy=force[i], idx=i)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            # self.save(data_list, processed_path)
            data, slices = self.collate(data_list)
            torch.save((data, slices), processed_path)

from equiformer.datasets.pyg.md17 import make_splits, train_val_test_split


def get_rmd17_datasets(
        root, dataset_arg, train_size, val_size, test_size, seed, 
        revised=False, 
        consecutive=False, 
        return_idx=False
    ):
    """
    Return training, validation and testing sets of MD17 with the same data partition as TorchMD-NET.
    """

    if revised:
        root = root.replace('md17', 'rmd17')

    # keys: ['z', 'pos', 'batch', 'y', 'dy']
    all_dataset = MD17(root, dataset_arg, revised=revised)

    idx_train, idx_val, idx_test = make_splits(
        len(all_dataset),
        train_size,
        val_size,
        test_size,
        seed,
        filename=os.path.join(root, "splits.npz"),
        splits=None,
        # idx are consecutive -> important for fixed-point reuse
        order='consecutive' if consecutive else None,
    )

    if return_idx:
        return idx_train, idx_val, idx_test

    train_dataset = Subset(all_dataset, idx_train)
    val_dataset = Subset(all_dataset, idx_val)
    test_dataset = Subset(all_dataset, idx_test)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":

    from torch_geometric.loader import DataLoader

    _root_path = "./datasets/test_md17/aspirin"
    train_dataset, val_dataset, test_dataset = get_rmd17_datasets(
        root=_root_path,
        dataset_arg="aspirin",
        train_size=950,
        val_size=50,
        test_size=None,
        seed=1,
        revised=True
    )

    print("Training set size:   {}".format(len(train_dataset)))
    print("Validation set size: {}".format(len(val_dataset)))
    print("Testing set size:    {}".format(len(test_dataset)))

    print('train_dataset[2]', train_dataset[2])

    train_loader = DataLoader(train_dataset, batch_size=8)
    for i, data in enumerate(train_loader):
        print('data', data)
        print('data.y', data.y)
        break
