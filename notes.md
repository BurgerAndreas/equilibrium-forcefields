

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
- Machine Learning of Accurate Energy-Conserving Molecular Force Fields
- use revised version!
- 10 small, organic molecules at DFT accuracy
- Energies and forces for molecular dynamics trajectories of eight organic molecules. Level of theory DFT: PBE+vdW-TS
- Each molecule has >100k samples, but we use only <1k samples for training
- https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.MD17.html
- total time of simulation was 200 ps for aspirin and 100 ps for the rest of the molecules, and for all the cases the NVT ensemble was used with a time step of 0.5 fs

revised
- 100,000 structures were randomly selected and redone
- original: aspirin has >200k samples

Equiformer uses the non-revised version: equiformer/datasets/pyg/md17.py

Are the units the same?
Nequip: [meV] and [meV/Å]
MACE: Energy (E, meV) and force (F, meV/Å)

md17: Units	Ang (length), kcal/mol (energy)

[revised MD17](https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038)
'nuclear_charges' : The nuclear charges for the molecule
'coords' : The coordinates for each conformation (in units of ångstrom)
'energies' : The total energy of each conformation (in units of kcal/mol)
'forces' : The cartesian forces of each conformation (in units of kcal/mol/ångstrom)
'old_indices' : The index of each conformation in the original MD17 dataset
'old_energies' : The energy of each conformation taken from the original MD17 dataset (in units of kcal/mol)
'old_forces' : The forces of each conformation taken from the original MD17 dataset (in units of kcal/mol/ångstrom)

Molecules@CCSD/CCSD(T): The data set of
small molecules at CCSD and CCSD(T) accuracy
[37] contains positions, energies, and forces for five
different small molecules: Asprin (CCSD), Benzene,
Malondaldehyde, Toluene, Ethanol (all CCSD(T)).
Each data set consists of 1,500 structures with the
exception of Ethanol, for which 2,000 structures are
available. For more detailed information, we direct
the reader to [37]. The data set was obtained from
http://quantum-machine.org/gdml/#datasets

### MD 22
- Accurate global machine learning force fields for molecules with hundreds of atoms
- http://www.sgdml.org/#datasets

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


## Others

Clifford Group Equivariant Networks (NeurIPS 2023 Oral)
- a three-dimensional n-body experiment, a four-dimensional Lorentz-equivariant high-energy physics experiment, and a five-dimensional convex hull experiment
- MLP-like architecture and use it in a message-passing graph network

Geometric Algebra Transformer (Neurips 2023)
- GATr is equivariant with respect to E(3)
- n-body modeling, wall-shear-stress estimation on large arterial meshes, robotic motion planning

Equivariant Flow Matching 
- two many-particle systems, DW4 and LJ13, pair-wise double-well and Lennard-Jones interactions with 4 and 13 particles
- LJ55 large Lennard-Jones cluster with 55 particles
- small molecule alanine dipeptide (Figure 3a) in Cartesian coordinates. The objective is to train a Boltzmann Generator capable of sampling from the equilibrium Boltzmann distribution defined by the semi-empirical GFN2-xTB force-field

E(n) equivariant normalizing flows
-

EGNN: E(n) equivariant graph neural networks

## Normalization

Self-Normalizing Neural Networks
- batch normalization evolved into a standard to normalize neuron activations to zero mean and unit variance [17]. Layer normalization [1] also ensures zero mean and unit variance, while weight normalization [25] ensures zero mean and unit variance if in the previous layer the activations have zero mean and unit variance.
- training with normalization techniques is perturbed by stochastic gradient descent (SGD), stochastic regularization (like dropout), and the estimation of the normalization parameters. 
- Both RNNs and CNNs can stabilize learning via weight sharing, therefore they are less prone to these perturbations
- SELUs push neuron activations to zero mean and unit variance
- For the weight initialization, we propose ω = 0 and τ = 1 for all units in the higher layer.
- for n units we define the mean of the weights as ω and the second moment as τ
- if (ω, τ ) is close to (0, 1), then the mean-and-variance-mapping-function g still has an attracting and stable fixed point that is close to (0, 1) 
- Dropout fits well to rectified linear units, since zero is in the low variance region and corresponds to the default value
- propose “alpha dropout”, that randomly sets inputs to α′, −λα = α′

### Norm

There are three important types of matrix norms.  For some matrix $A$
- Induced norm, which measures what is the maximum of $\frac{\|Ax\|}{\|x\|}$ for any $x \neq 0$ (or, equivalently, the maximum of $\|Ax\|$ for $\|x\|=1$).
- Element-wise norm, which is like unwrapping $A$ into a long vector, then calculating its vector norm.
- Schatten norm, which measures the vector norm of the singular values of $A$.

Equivalences:
- Frobenius norm = Element-wise 2-norm = Schatten 2-norm
- Induced 2-norm = Schatten $\infty$-norm.  This is also called Spectral norm.

https://stats.stackexchange.com/questions/467654/relation-between-frobenius-norm-and-l2-norm

For vectors:
- L1 = Manhatten norm = sum of absolute values
- L2 = Euclidean norm = shortest path = square root of sum of squares
- Lp = p-norm = root of sum of p-th power of elements
- L-infinity = max norm = max of absolute values

- torch.norm(input, p='fro', dim=None)
- torch.linalg.norm(A, ord=None, dim=None)

### Initialization

- The default initialization for a Linear layer is init.kaiming_uniform_(self.weight, a=math.sqrt(5))

Energy as gradient of the force via @torch.enable_grad() of forward w.r.t pos (see dp_attention_transformer_md17.py)
```python
@torch.enable_grad()
forward():
    pos = pos.requires_grad_(True)
    # ...
    z_pred, info = self.deq(...) # deq
    # ...
    forces = -1 * (
        torch.autograd.grad(
            energy,
            pos,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
        )[0]
    )
```

- Is the force gradient exact? 
-> If grad=1 (default), forces use approximate 1-step phantom grad from TorchDEQ for autodiff

- exact gradient with torch autograd
- - [x] default: `deq_kwargs.grad=1` Phantom Gradient (running 1 function step at the fixed point and differentiating it as the backward pass)
- - [x] `deq_kwargs.ift=True` or `deq_kwargs.grad='ift'` for exact implicit differentiation
- - [x] `deq_kwargs.grad=10 deq_kwargs.f_max_iter=0` for exact Back Progapagation Through Time (BPTT) with PyTorch autograd (should give the same result as IFT, use as a sanity check)

- equiformer equivariance test (model(x) == model(rot(x)))


## Normalization

TorchDEQ
- Weight Normalization (WN) scales the L2 norm using a learnable scaling factor
- Spectral Normalization (SN) divides the weight matrix by its spectral norm, usually computed by power iterations
- gradient clipping significantly stabilizes the training of implicit graph neural networks [29] on node classification, i.e., by clipping the rescaling factor f to a threshold t
Regularization
- jac_reg penalizes the upper bound of Jacobian spectral radius
- for transformer, found that Pre Layer Norm (LN) can stabilize the DEQ transformer under a higher gradient step than Post LN
- uses ReLU, does not find a significant difference between activation functions

Stabilizing Equilibrium Models by Jacobian Regularization
- explicitly regularizes the Jacobian of the fixed-point update equations 
- stabilizes the learning of equilibrium models
- significantly stabilizes the fixed-point convergence in both forward and backward passes
- adds only minimal computational cost, scales well to high-dimensional domains
- additional soft loss term, normalize spectral norm of Jacobian via the Frobenius norm, estimated via M=1 Monte Carlo estimate of Hutchinson estimator
- the larger the batch_size, the lower M can be chosen (batch_size=64 to 128, M=1)
Growing instability during training
- DEQ at the end of training can take > 3× more iterations
Regular NN regularizations used in DEQ
- weight normalization (Salimans & Kingma, 2016), recurrent dropout (Gal & Ghahramani, 2016), and group normalization (Wu & He, 2018)

Deep equilibrium networks are sensitive to initialization statistics
- initializing with orthogonal or symmetric matrices allows for greater stability in training


https://implicit-layers-tutorial.org/deep_equilibrium_models/
-  In pratice, it’s also important to apply some form of normalization before and after the DEQ layer: here we simply use Batch Norm

One-Step Diffusion Distillation via Deep Equilibrium Models
- image Decoder: The output of the Generative Equilibrium Transformer (GET)-DEQ is first normalized with Layer Normalization
- for input injection adaptive layer normalization (AdaLN-Zero) performs worse than additive injection
- https://github.com/locuslab/get/blob/53377d3047b8ab677ef4e84252b15b5919fe70dd/models/get.py#L143

Exploiting Connections between Lipschitz Structures for Certifiably Robust Deep Equilibrium Models
- various popular Lipschitz network structures -> reparametrize as Lipschitz-bounded equilibrium networks (LBEN) -> certifiably robust


## Literature

- On Training Implicit Models
Phantom Gradients
