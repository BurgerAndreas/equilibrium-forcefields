

Equiformer
- 6 blocks 
last block is different (changes shape of node_features)
- Lmax = 2
- Gaussian radial basis

- equivariant graph attention improves upon typical dot product attention
- Non-linear message passing almost twice as slow as MLP attention or Dot product
- MLP attention is roughly the same as dot product attention, depends on the task
- QM9: one A6000 GPU with 48GB, one epoch takes ~ 10 minutes
- training time: QM9, MD17 << OC20


Equiformer v2
- replace SO(3) convolutions with eSCN convolutions to efficiently incorporate higher-degree tensors
- to better leverage the power of higher degrees, we propose three architectural improvements -- attention re-normalization, separable S2 activation and separable layer normalization
- eval on OC20 only
- On OC20 for Lmax = 2, EquiformerV2 is 2.3x faster than Equiformer since EquiformerV2 uses eSCN convolutions for efficient SO(3) convolutions
- On QM9 the performance gain is not so significant, since QM9 contains only 110k examples, which is much less than OC20 S2EF-2M with 2M examples

TorchDEQ
- the zeroth-order approximation, i.e. 1-step gradient, is the default backpropagation option (grad=1)


deq_dot_product_attention_transformer_exp_l2_md17
batch_size=1, num_layers=2, RTX3060, target=aspiring ~ 170s per epoch -> 1k epochs = 48 hours = 2 days

## Datasets
see Allegro paper for great summary

### MD 17
- use revised version!
- 10 small, organic molecules at DFT accuracy
- Energies and forces for molecular dynamics trajectories of eight organic molecules. Level of theory DFT: PBE+vdW-TS
- Each molecule has >100k samples, but we use only <1k samples for training
- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MD17.html
- Machine learning of accurate energy-conserving molecular force fields

### QM9
- 133,885 structures with up to 9 heavy elements and consisting of species H, C, N, O, F in relaxed geometries. Structures are provided together with a series of properties computed at the DFT/B3LYP/6-31G(2df,p) level of theor


## E3NN

"128x0e+64x1e+32x2e"
oe.Irreps() -> 128*1 + 64*3 + 32*5 = 480

128x0e+64x1e+32x2e+64x0e+32x1e+16x2e

for group element g ∈ SO(3), there are (2L+1)-by-(2L+1) irreps matrices DL(g) called Wigner-D matrices 
acting on (2L + 1)-dimensional vectors

concatenate multiple type-L vectors to form SE(3)-equivariant irreps features

feature = flatten([
    (l0^1), (l0^2), ..., 
    (l1^1_x, l1^1_y, l1^1_z), (l1^2_x, l1^2_y, l1^2_z), ..., 
    (l2^1_a, l2^1_b, l2^1_c, l2^1_d, l2^1_e), (l2^2_a, l2^2_b, l2^2_c, l2^2_d, l2^2_e), ...
])



## Allegro
- https://github.com/mir-group/allegro
- https://github.com/mir-group/nequip
- Large-scale Atomic/Molecular Massively Parallel Simulator (LAMMPS) Molecular Dynamics Simulator
- free and open-source software

Does not directly implement any datasets?

How to incorporate the datasets:
- https://github.com/mir-group/nequip/discussions/305#discussioncomment-5056437
- https://nequip.readthedocs.io/en/latest/howto/dataset.html#prepare-dataset-and-specify-in-yaml-config

1layer and 3-layer Allegro networks have 7,375,237 and 17,926,533 parameters