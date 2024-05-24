from math import sqrt

lr = 5e-4
bs = 4
b1 = 0.9
b2 = 0.999
eps = 1e-8

new_bs = 8
k = new_bs / bs
new_lr = lr * sqrt(k)
new_b1 = 1 - k * (1 - b1)
new_b2 = 1 - k * (1 - b2)
new_eps = eps / sqrt(k)

print(f"new_lr: {new_lr}")
print(f"new_b1: {new_b1}")
print(f"new_b2: {new_b2}")
print(f"new_eps: {new_eps}")


def scale_batchsize_lr(args, k=None):
    if k is not None:
        args.batch_size = args.batch_size * k
        args.lr = args.lr * sqrt(k)
        betas = args.get("opt_betas", None)
        if betas is None:
            betas = [0.9, 0.99]
        args.opt_betas = [1 - k * (1 - betas[1]), 1 - k * (1 - betas[2])]
        opt_eps = args.get("opt_eps", 1e-8)
        args.opt_eps = opt_eps / sqrt(k)
    return args
