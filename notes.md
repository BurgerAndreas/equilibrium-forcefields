

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

Md17
- Energies and forces for molecular dynamics trajectories of eight organic molecules. Level of theory DFT: PBE+vdW-TS
- Each molecule has >100k samples, but we use <1k samples for training
- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MD17.html
- Machine learning of accurate energy-conserving molecular force fields

