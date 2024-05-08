import torch

import torchdeq
from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm

"""
What happens if we set f_max_iter=0?
All steps are done with autograd?


"""

# set everything to GPU
torch.set_default_tensor_type(torch.cuda.FloatTensor)
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

def run_deq(deq, f, name=""):
    z0 = torch.tensor(0.0)
    z_out, info = deq(f, z0)

    print(f'nstep', info['nstep'])

    tgt = torch.tensor(0.5)
    loss = (z_out[-1] - tgt).abs().mean()
    loss.backward()

    print(f'Loss & Grad {name}:', loss.item(), theta.grad)


# """ Define function """
# # Input Injection & Equilibrium function
# x = torch.tensor(1.0)
# theta = torch.tensor(0.0, requires_grad=True)
# f = lambda z: torch.cos(z) + theta


# """ 1-step grad """
# print("\n1-step grad")
# # broyden, fixed_point_iter
# deq = get_deq(f_solver='fixed_point_iter', f_max_iter=20)
# run_deq(deq, f)


# """ BPTT, 10 steps """
# print("\nBPTT, 10 steps")
# deq = get_deq(grad=10, f_max_iter=0)
# run_deq(deq, f)


# """ BPTT, 100 steps """
# print("\nBPTT, 100 steps")
# deq = get_deq(grad=100, f_max_iter=0)
# run_deq(deq, f)


""" Broyden with gradients """
print("\nBroyden with gradients")
from torchdeq.solver.broyden import broyden_solver, broyden_solver_grad

# broyden_solver(func, x0, max_iter=50, tol=1e-3, stop_mode="abs", indexing=None, LBFGS_thres=None, ls=False, return_final=False)
# deq = lambda f, z0: broyden_solver(f, z0, max_iter=20, tol=1e-3)
def broyden_wrapper(func, z0):
    z_out, _, info = broyden_solver(func=func, x0=z0, max_iter=20, tol=1e-3)
    return z_out, info 

# deq = broyden_wrapper
# run_deq(deq, f, name="Broyden with gradients")


torch.autograd.set_detect_anomaly(True)

theta = torch.tensor(0.0, requires_grad=True)
f = lambda z: torch.cos(z) + theta

z0 = torch.tensor(0.0)
z0.requires_grad = True
# This will only work for max_iter=1
z_out, _, info = broyden_solver_grad(f, z0, max_iter=4, tol=1e-3, return_final=False)
# z_out = [z_out]

print(f'nstep', info['nstep'])
print(f'z_out', z_out.requires_grad)

tgt = torch.tensor(0.5)
loss = (z_out - tgt).abs().mean()
loss.backward()

print(f'Loss & Grad:', loss.item(), theta.grad)
print(f'nstep', info['nstep'])


# # not differentiable:
# foo[mask] = bar  # mask is a boolean tensor

# # differentiable:
# update = torch.zeros_like(foo)  # doesn't require grad
# update[mask] = bar # because update didn't require grad, this works with autograd
# foo = foo + update  # this is differentiable

_f = 2
print(str(float(_f)))