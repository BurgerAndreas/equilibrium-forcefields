import torch
from e3nn import o3
from torch_geometric.data import Data
import equiformer_v2
from equiformer_v2.nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

# https://github.com/atomicarchitects/equiformer_v2/issues/5

# number of nodes=atoms
n = 10

edge_index = torch.tensor(
    [[0, 1, 1, 2],
    [1, 0, 2, 1]], dtype=torch.long
)
pos = torch.randn(n, 3)
sample = Data(
    pos=pos, edge_index=edge_index,
    atomic_numbers=torch.randint(0, 20, (n,))
)

R = torch.tensor(o3.rand_matrix())

model = EquiformerV2_OC20(
    num_layers=2,
    attn_hidden_channels=16,
    ffn_hidden_channels=16,
    sphere_channels=16,
    edge_channels=16,
    alpha_drop=0.0, # Turn off dropout for eq
    drop_path_rate=0.0, # Turn off drop path for eq
)

energy1, forces1 = model(sample)

# rotate molecule
rotated_pos = torch.matmul(pos, R)
sample.pos = rotated_pos
energy_rot, forces_rot = model(sample)

print(
    # energy should be invariant
    energy1 == energy_rot,
    # forces should be equivariant
    # model(rot(f)) == rot(model(f))
    torch.allclose(torch.matmul(forces1, R), forces_rot, atol=1.0e-3),
    (torch.matmul(forces1, R) - forces_rot).abs().max()
)