# equilibrium-forcefields
Scaling Force Fields with Equivariant Deep Equilibrium Models

Perks of Equilibrium Models:
- memory-efficient backpropagation via implicit differentiation -> less memory use -> larger batch sizes

## Quickstart

```bash
python scripts/deq_equiformer.py
# baseline equiformer
python equiformer/main_md17.py 
# for revised MD17 dataset: 
python scripts/deq_equiformer.py md17revised=True
# for the model version used in the equiformer paper:
python scripts/deq_equiformer.py model_name="graph_attention_transformer_nonlinear_l2_md17"
# all args as in the equiformer paper:
python scripts/deq_equiformer.py +preset=md17_aspirin_sel2
```

On a slurm cluster:
```bash
sbatch scripts/slurm_launcher.slrm deq_equiformer.py
# V1
sbatch scripts/slurm_launcher.slrm deq_equiformer.py model_kwargs.input_injection=False 
# baseline equiformer
sbatch scripts/slurm_launcher.slrm main_md17.py num_layers=2
```

### EquiformerV2

If there is an import error, it is possible that [`ocp/ocpmodels/common/utils.py`](https://github.com/Open-Catalyst-Project/ocp/blob/5a7738f9aa80b1a9a7e0ca15e33938b4d2557edd/ocpmodels/common/utils.py#L329) is not modified. 
Please follow [here](equiformer_v2/docs/env_setup.md) for details.

#### New (Hydra)

Baseline Equiformer V2
```bash
python /ssd/gen/equilibrium-forcefields/scripts/deq_equiformer_v2.py
sbatch scripts/slurm_launcher.slrm deq_equiformer_v2.py
```

DEQ Equiformer V2
```bash
python /ssd/gen/equilibrium-forcefields/scripts/deq_equiformer_v2.py ++use=deq ++preset=small_l6
sbatch scripts/slurm_launcher.slrm deq_equiformer_v2.py ++use=deq
```

#### Old (argparse+yml)

Baseline Equiformer V2
```bash
/home/andreasburger/miniforge3/envs/deq/bin/python /ssd/gen/equilibrium-forcefields/equiformer_v2/main_oc20.py --mode train --config-yml 'equiformer_v2/oc20/configs/s2ef/2M/equiformer_v2_tiny_l6.yml'
# or
source equiformer_v2/scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_small_l3.sh
# slurm cluster
sbatch scripts/slurm_launcher_v2_argparse.slrm base small_l3
```

DEQ Equiformer V2
```bash
/home/andreasburger/miniforge3/envs/deq/bin/python /ssd/gen/equilibrium-forcefields/equiformer_v2/main_oc20.py --mode train --config-yml 'equiformer_v2/oc20/configs/s2ef/2M/deq_equiformer_v2_tiny_l6.yml' optim.batch_size=2
# or
source equiformer_v2/scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_small_l3.sh
# slurm cluster
sbatch scripts/slurm_launcher_v2_argparse.slrm deq small_l3
```

#### Original
```bash
# OC20 **S2EF-2M**, 2 nodes with 8 GPUs on each node.
# sh scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@12_L@6_M@2_splits@2M_g@multi-nodes.sh
python main_oc20.py \
    --distributed \
    --num-gpus 8 \
    --num-nodes 2 \
    --mode train \
    --config-yml 'oc20/configs/s2ef/2M/equiformer_v2/equiformer_v2_N@12_L@6_M@2.yml' \
    --run-dir 'models/oc20/s2ef/2M/equiformer_v2/N@12_L@6_M@2/bs@64_lr@2e-4_wd@1e-3_epochs@12_warmup-epochs@0.1_g@8x2' \
    --print-every 200 \
    --amp \
    --submit


# EquiformerV2 (31M)** on OC20 **S2EF-All+MD**
# sh equiformer_v2/scripts/train/oc20/s2ef/equiformer_v2/equiformer_v2_N@8_L@4_M@2_splits@all+md_g@multi-nodes.sh
python main_oc20.py \
    --distributed \
    --num-gpus 8 \
    --num-nodes 8 \
    --mode train \
    --config-yml 'oc20/configs/s2ef/all_md/equiformer_v2/equiformer_v2_N@8_L@4_M@2_31M.yml' \
    --run-dir 'models/oc20/s2ef/all_md/equiformer_v2/N@8_L@4_M@2_31M/bs@512_lr@4e-4_wd@1e-3_epochs@3_warmup-epochs@0.01_g@8x8' \
    --print-every 200 \
    --amp \
    --submit
```

## TODO

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


- max out GPU memory via network parameters \
Yes, sounds important

- find optimal hyperparameters (at least learning_rate)

- max out GPU memory via batch_size \
Maybe, doesn't seem too important

- [x] Try anderson and broyden solver (default: fixed_point_iter) \
In theory, if all methods find a fixed point within `f_max_iter`, all should converge to the same fixed point. Then the method should not affect the error of the DEQ over number of steps.

- [x] Only calculate the loss w.r.t energy \
meas_force=False

- Is the force gradient exact? 
-> If grad=1 (default), forces use approximate 1-step phantom grad from TorchDEQ for autodiff


- think about fix point reuse (not sure if the MD dataset is temporally ordered. At least need to change shuffeling and batching in the dataloader)
Not too important right now since it only increases speed in time but not nimber of epochs and it will be difficult to implement

- exact gradient with torch autograd
- - [x] default: `deq_kwargs.grad=1` Phantom Gradient (running 1 function step at the fixed point and differentiating it as the backward pass)
- - [x] `deq_kwargs.ift=True` or `deq_kwargs.grad='ift'` for exact implicit differentiation
- - [x] `deq_kwargs.grad=10 deq_kwargs.f_max_iter=0` for exact Back Progapagation Through Time (BPTT) with PyTorch autograd (should give the same result as IFT, use as a sanity check)

- [x] equiformer num_layers=2 fuer vergleichbarkeit

- fixed point error (do we converge to a fixed point)?
- - Broyden solver NaN: numerical instability?
- - `f_stop_mode='rel'` or `'abs'`? set `deq_kwargs.f_max_iter=100 deq_kwargs.b_max_iter=100`

- DEQ torch norm tricks (e.g. see https://github.com/locuslab/torchdeq/blob/main/deq-zoo/ignn/graphclassification/layers.py)

- [x] exploding weights, gradients, activations?
- [ ] how should the weights, gradients, activations look like?

- equiformer equivariance test (model(x) == model(rot(x)))


## Normalization

- [ ] Jacobian regularization

Equiformer


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