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
- force weglassen, nur energy in loss
- fixed point error (do we converge to a fixed point)?
- - Broyden solver NaN: numerical instability?
- - f_stop_mode='rel' (default) or 'abs'?
- Broyden solver NaN: numerical instability?
- DEQ paper: use of (layer)norm?
- DEQ torch norm tricks
- exploding weights or activations?
- equformer num_layers=2 fuer vergleichbarkeit
- equiformer equivariance test (model(x) == model(rot(x)))
- initalize z0=0 (why do others do it and how can we do it)


## Experiments to run

Try exact gradients
deq_kwargs.ift=True

deq_kwargs.grad=2


Only calculate the loss w.r.t energy