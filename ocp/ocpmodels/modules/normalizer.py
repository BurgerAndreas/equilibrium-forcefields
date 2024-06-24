"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch


class Normalizer(object):
    """Normalize a Tensor and denorm it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        """
        Usage:
            normalizer = Normalizer(tensor)
            # or
            normalizer = Normalizer(mean=mean, std=std)
        """
        if tensor is None and mean is None:
            return

        if device is None:
            device = "cpu"
        self.device = device

        # compute mean and std from tensor
        if tensor is not None:
            self.mean = torch.mean(tensor, dim=0).to(device)
            self.std = torch.std(tensor, dim=0).to(device)
            # print(
            #     f"{self.__class__.__name__} computed mean: {self.mean}, std: {self.std}"
            # )
            return

        if (mean is not None) and (std is not None):
            self.mean = torch.tensor(mean).to(device)
            self.std = torch.tensor(std).to(device)
            # print(
            #     f"{self.__class__.__name__} loaded mean: {self.mean}, std: {self.std}"
            # )

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
    
    def __call__(self, *args, **kwds):
        return self.norm(*args, **kwds)
    
    def norm(self, tensor, z=None):
        """_summary_

        Args:
            tensor (_type_): _description_
            z (_type_, optional): atom types. Is ignored. Defaults to None.

        Returns:
            _type_: _description_
        """
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor, z=None):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].to(self.mean.device)
        self.std = state_dict["std"].to(self.mean.device)

class NormalizerByAtomtype(Normalizer):
    """Normalize a Tensor by atom type and denorm it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        super().__init__(tensor, mean, std, device)
        assert len(self.mean.shape) == 1, self.mean.shape # [Atoms]
        assert self.mean.shape == self.std.shape

    def norm(self, tensor, z=None):
        """tensor: [N, 3], z: [N]"""
        return (tensor - self.mean[z].unsqueeze(1)) / self.std[z].unsqueeze(1)

    def denorm(self, normed_tensor, z=None):
        """tensor: [N, 3], z: [N]"""
        return normed_tensor * self.std[z].unsqueeze(1) + self.mean[z].unsqueeze(1)
    
class NormalizerByAtomtype3D(Normalizer):
    """Normalize a Tensor componentwise and by atom type, denorm it later."""

    def __init__(self, tensor=None, mean=None, std=None, device=None):
        super().__init__(tensor, mean, std, device)
        assert len(self.mean.shape) == 2, self.mean.shape # [Atoms]
        assert self.mean.shape == self.std.shape

    def norm(self, tensor, z=None):
        """tensor: [N, 3], z: [N]"""
        return (tensor - self.mean[z]) / self.std[z]

    def denorm(self, normed_tensor, z=None):
        """tensor: [N, 3], z: [N]"""
        return normed_tensor * self.std[z] + self.mean[z]
