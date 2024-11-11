import torch

drop = torch.nn.Dropout(p=0.5)

x = torch.ones(2, 3, 4)
# x = torch.ones(2, 1, 4, 1)

# for _ in range(10):
#     mask = torch.ones((2, 1, 4), dtype=x.dtype, device=x.device)
#     mask = drop(mask)
#     out = x * mask
#     print(out)
#     print(out.shape)
#     print()

for _ in range(10):
    print()
    x = torch.ones(2, 3, 4)
    x = drop(x)
    print(x)
