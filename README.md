# equilibrium-forcefields
Scaling Force Fields with Equivariant Deep Equilibrium Models

Perks of Equilibrium Models:
- memory-efficient backpropagation via implicit differentiation -> less memory use -> larger batch sizes

## Quickstart

```bash
python scripts/deq_equiformer.py
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

max out GPU memory via batch_size
1 Können wor versuchen, aber ist glaube ich nicht das wichtigste

Try anderson and broyden solver (now: fixed_point_iter)
2. Das würde am der convergence in number of epochs nichts ändern, da die ja alle zum gleichen fixpoint iteroeren sollen

Try exact gradients or only calculate the loss w.r.t energy?
3. Exact gradients ist wichtig zu testen, auch der Vergleich zu backprop un sicher zu stellen, dass alles richtig funktioniert

Ist der force gradient exakt? 
-> yes (see dp_attention_transformer_md17.py)

find optimal hyperparameters (at least learning_rate)
4. Jo, das ist definitiv wichtig

think about fix point reuse (not sure if the MD dataset is temporally ordered. At least need to change shuffeling and batching in the dataloader)
5. Not too important right now since it only increases speed in time but not nimber of epochs and it will be difficult to implement

max out GPU memory via network parameters
6. Jo, das klingt auch viel versprechend


## Experiments to run

Try exact gradients
deq_kwargs.ift=True

deq_kwargs.grad=2


Only calculate the loss w.r.t energy