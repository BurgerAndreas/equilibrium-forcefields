import torch
from torch_geometric.data import InMemoryDataset, download_url, Data, extract_tar, extract_zip
import numpy as np
from torch.utils.data import Subset
import os
import os.path as osp


######################################################################################
# Revised MD17 dataset
# https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038
# 100k samples

filepath = 'datasets/rmd17/aspirin/raw/rmd17/npz_data/rmd17_aspirin.npz'
raw_data = np.load(filepath)

z = torch.from_numpy(raw_data['nuclear_charges']).long()
pos = torch.from_numpy(raw_data['coords']).float()
# https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038
# 'old_indices' : The index of each conformation in the original MD17 dataset
# 'old_energies' : The energy of each conformation taken from the original MD17 dataset (in units of kcal/mol)
# 'old_forces': The forces of each conformation taken from the original MD17 dataset (in units of kcal/mol/ångstrom)
energy = torch.from_numpy(raw_data['energies']).float()
energy_old = torch.from_numpy(raw_data['old_energies']).float()
force_old = torch.from_numpy(raw_data['old_forces']).float()
old_indices = torch.from_numpy(raw_data['old_indices']).long()

print(f'length data: {round(pos.size(0) / 1e3)}k')
print(f'pos size: {pos.size()}')
# print(z[0])
# print(pos[0])

print(f'energy: {energy[0]} (type {type(energy[0])})')
print(f'energy old: {energy_old[0]}  (type {type(energy[0])})')
print(f'old_indices: {old_indices[0]}  (type {type(old_indices[0])})')


print('')
print('-' * 80)
print('')
######################################################################################
# Original MD17 dataset
# 211,762 samples

filepath = 'datasets/md17/aspirin/raw/md17_aspirin.npz'
raw_data = np.load(filepath)

z = torch.from_numpy(raw_data['z']).long()
pos = torch.from_numpy(raw_data['R']).float()
energy = torch.from_numpy(raw_data['E']).float()
force = torch.from_numpy(raw_data['F']).float()

print(f'length data: {round(pos.size(0) / 1e3)}k')
print(f'pos size: {pos.size()}')
# print(z[0])
# print(pos[0])

print(f'energy: {energy[0]}  (type {type(energy[0])})')

print(f'energy old_indices: {energy[int(old_indices[0])]} ')