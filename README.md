# equilibrium-forcefields
Scaling Force Fields with Equivariant Deep Equilibrium Models

memory-efficient backpropagation via implicit differentiation
larger batch sizes

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


## Experiments to run

dot_product_attention_transformer_exp_l2_md17 with num_layers=6 and num_layers=2 and compare to deq with num_layers=2