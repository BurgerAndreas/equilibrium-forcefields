
import torch

import torchdeq
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = torch.nn.Linear(10, 10)

        # deq
        self.deq = get_deq()
        apply_norm(self.layer, 'weight_norm')

    def implicit_layer(self, x):
        return self.layer(x)
    
    def forward(self, x, pos):

        z = torch.zeros_like(x)

        reset_norm(self.layer)

        f = lambda z: self.f(z, pos)

        z_pred, info = self.deq(self.implicit_layer, z)
                                
        energy = z_pred[-1]
        forces = -1 * (
            torch.autograd.grad(
                energy,
                # diff with respect to pos
                # if you get 'One of the differentiated Tensors appears to not have been used in the graph'
                # then because pos is not 'used' to calculate the energy
                pos, 
                grad_outputs=torch.ones_like(energy),
                create_graph=True,
                # allow_unused=True, # TODO
            )[0]
        )

        return energy, forces


def run(model, eval=False):

    if eval:
        model.eval()
    else:
        model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(10):
        x = torch.randn(10, 10)
        pos = torch.randn(10, 3)
        energy, forces = model(x, pos)
        
        # loss
        optimizer.zero_grad()
        energy_target = torch.randn(10, 1)
        energy_loss = torch.nn.functional.mse_loss(energy, energy_target)
        force_target = torch.randn(10, 3)
        force_loss = torch.nn.functional.mse_loss(forces, force_target)
        loss = energy_loss + force_loss

        if not eval:
            loss.backward()
            optimizer.step()
    
    return True

if __name__ == '__main__':
    model = MyModel()
    success = run(model, eval=False)
    print(f'train success: {success}')
    success = run(model, eval=True)
    print(f'eval success: {success}')