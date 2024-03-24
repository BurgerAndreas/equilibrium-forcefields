

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

the zeroth-order approximation, i.e., 1-step gradient, as the default backpropagation option in TorchDEQ
grad=1


deq_dot_product_attention_transformer_exp_l2_md17
batch_size=1, num_layers=2, RTX3060, target=aspiring ~ 170s per epoch -> 1k epochs = 48 hours = 2 days