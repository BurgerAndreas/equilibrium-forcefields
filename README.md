# equilibrium-forcefields
Scaling Force Fields with Equivariant Deep Equilibrium Models

Perks of Equilibrium Models:
- memory-efficient backpropagation via implicit differentiation -> less memory use -> larger batch sizes

## Quickstart

```bash
python scripts/deq_equiformer.py
# baseline equiformer
python equiformer/main_md17.py num_layers=2
```

On a slurm cluster:
```bash
sbatch scripts/slurm_launcher.slrm deq_equiformer.py
sbatch scripts/slurm_launcher.slrm deq_equiformer.py z_is_node_features=True # V1
# sbatch scripts/slurm_launcher.slrm main_md17.py num_layers=2
```

## TODO

- [ ] Energy as gradient of the force via @torch.enable_grad() of forward w.r.t pos (see dp_attention_transformer_md17.py)
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

- max out GPU memory via batch_size \
Können wor versuchen, aber ist glaube ich nicht das wichtigste

- Try anderson and broyden solver (now: fixed_point_iter) \
Das würde am der convergence in number of epochs nichts ändern, da die ja alle zum gleichen fixpoint iteroeren sollen

Try exact gradients or only calculate the loss w.r.t energy?
Exact gradients ist wichtig zu testen, auch der Vergleich zu backprop um sicher zu stellen, dass alles richtig funktioniert

- Ist der force gradient exakt? 
-> yes? (see dp_attention_transformer_md17.py). Does torch use deq grad for autodiff?

- find optimal hyperparameters (at least learning_rate)
Jo, das ist definitiv wichtig

- think about fix point reuse (not sure if the MD dataset is temporally ordered. At least need to change shuffeling and batching in the dataloader)
Not too important right now since it only increases speed in time but not nimber of epochs and it will be difficult to implement

- max out GPU memory via network parameters \
Jo, das klingt auch viel versprechend

- exact gradient with torch autograd
- - [x] default: `deq_kwargs.grad=1` Phantom Gradient (running 1 function step at the fixed point and differentiating it as the backward pass)
- - [x] `deq_kwargs.ift=True` or `deq_kwargs.grad='ift'` for exact implicit differentiation
- - [x] `deq_kwargs.grad=10 deq_kwargs.f_max_iter=0` for exact Back Progapagation Through Time (BPTT) with PyTorch autograd (should give the same result as IFT, use as a sanity check)

- [x] ignore force, only energy in loss\ 
meas_force=False

- equiformer num_layers=2 fuer vergleichbarkeit

- fixed point error (do we converge to a fixed point)?
- - Broyden solver NaN: numerical instability?
- - `f_stop_mode='rel'` or `'abs'`? set `deq_kwargs.f_max_iter=100 deq_kwargs.b_max_iter=100`

- Broyden solver NaN: numerical instability?

- DEQ paper: use of (layer)norm?

- DEQ torch norm tricks (e.g. see https://github.com/locuslab/torchdeq/blob/main/deq-zoo/ignn/graphclassification/layers.py)

- exploding weights, gradients, activations?
- [ ] how should the weights, gradients, activations look like?

- equiformer equivariance test (model(x) == model(rot(x)))

- initalize z0=0 (why do others do it and how can we do it)



## Experiments to run

Try exact gradients


Only calculate the loss w.r.t energy