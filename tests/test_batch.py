import torch

shape_per_sample = [21, 16, 64]
shape_per_sample = [2, 3, 5]
batch_size = 4

fps = [
    torch.full(shape_per_sample, 0, dtype=torch.float32),
    torch.full(shape_per_sample, 1, dtype=torch.float32),
    torch.full(shape_per_sample, 2, dtype=torch.float32),
    torch.full(shape_per_sample, 3, dtype=torch.float32),
]

indices = [0, 1, 2, 3]

fp = torch.cat([fps[_idx] for _idx in indices], dim=0)

# [84, 16, 64] -> [4, 21, 16, 64]
fp_undone = fp.view(batch_size, -1, *fp.shape[1:])
print("fp_undone:", fp_undone.shape)

fps_undone = [None] * batch_size
for _idx, _fp in zip(indices, fp_undone):
    fps_undone[_idx] = _fp

for _fp, _fp_undone in zip(fps, fps_undone):
    assert torch.equal(_fp, _fp_undone), (
        f"\n_fp: {_fp.shape}\n_fp_undone: {_fp_undone.shape}"
        f"\n_fp: {_fp}\n_fp_undone: {_fp_undone}"
    )
