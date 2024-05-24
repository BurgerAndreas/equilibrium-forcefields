"""
    Add `extra_repr` into PathDrop implemented by timm 
    for displaying more info.
"""


import torch
import torch.nn as nn
from e3nn import o3
import torch.nn.functional as F


def path_drop(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    Sample-wise stochastic depth is a regularization technique for networks with residual connections that probabilistically drops samples after the transformation function in each residual block.
    "stochastic depth" means dropping the entire conv and only keeping the residual.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class PathDrop(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(PathDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return path_drop(x, self.drop_prob, self.training)

    def extra_repr(self):
        return "drop_prob={}".format(self.drop_prob)


class GraphPathDrop(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Sample-wise stochastic depth is a regularization technique for networks with residual connections that probabilistically drops samples after the transformation function in each residual block.
    "stochastic depth" means dropping the entire conv and only keeping the residual.
    Consider batch for graph data when dropping paths.
    """

    def __init__(self, drop_prob=None):
        super(GraphPathDrop, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, batch):
        batch_size = batch.max() + 1
        shape = (batch_size,) + (1,) * (
            x.ndim - 1
        )  # work with diff dim tensors, not just 2D ConvNets
        ones = torch.ones(shape, dtype=x.dtype, device=x.device)
        drop = path_drop(ones, self.drop_prob, self.training)
        out = x * drop[batch]
        return out

    def extra_repr(self):
        return "drop_prob={}".format(self.drop_prob)


class EquivariantDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantDropout, self).__init__()
        self.irreps = irreps
        self.num_irreps = irreps.num_irreps
        self.drop_prob = drop_prob
        self.drop = torch.nn.Dropout(drop_prob, True)
        self.mul = o3.ElementwiseTensorProduct(
            irreps, o3.Irreps("{}x0e".format(self.num_irreps))
        )

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        shape = (x.shape[0], self.num_irreps)
        mask = torch.ones(shape, dtype=x.dtype, device=x.device)
        mask = self.drop(mask)
        out = self.mul(x, mask)
        return out


class EquivariantScalarsDropout(nn.Module):
    def __init__(self, irreps, drop_prob):
        super(EquivariantScalarsDropout, self).__init__()
        self.irreps = irreps
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        out = []
        start_idx = 0
        for mul, ir in self.irreps:
            temp = x.narrow(-1, start_idx, mul * ir.dim)
            start_idx += mul * ir.dim
            if ir.is_scalar():
                temp = F.dropout(temp, p=self.drop_prob, training=self.training)
            out.append(temp)
        out = torch.cat(out, dim=-1)
        return out

    def extra_repr(self):
        return "irreps={}, drop_prob={}".format(self.irreps, self.drop_prob)
