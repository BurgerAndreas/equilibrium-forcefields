import torch

# import necessary libraries
import torch

# define a tensor
A = torch.tensor(1.0, requires_grad=True)
print("Tensor A:", A)

# define a function using A tensor
def f(x):
    B = x + 1
    # check gradient
    print("B.requires_grad:", B.requires_grad)


""" Requires grad """
f(A)


""" Does not require grad """
with torch.no_grad():
    f(A)


""" Does not require grad """
test_w_no_grad = True
with torch.set_grad_enabled(not test_w_no_grad):
    f(A)


""" Does not require grad """
from contextlib import ExitStack

with ExitStack() as stack:
    if test_w_no_grad:
        stack.enter_context(torch.no_grad())
    f(A)
