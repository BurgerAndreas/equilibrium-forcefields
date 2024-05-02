import torch
import torch.nn as nn

# from ocpmodels.modules.loss import DDPLoss, L2MAELoss
# from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/modules/loss.py#L7
class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


# from equiformer/oc20/trainer/base_trainer_oc20.py
def load_loss(loss_fn={"energy": "mae", "force": "mae"}):
    for loss, loss_name in loss_fn.items():
        if loss_name in ["l1", "mae"]:
            loss_fn[loss] = nn.L1Loss()
        elif loss_name == "mse":
            loss_fn[loss] = nn.MSELoss()
        elif loss_name == "l2mae":
            loss_fn[loss] = L2MAELoss()
        else:
            raise NotImplementedError(f"Unknown loss function name: {loss_name}")
        # if distutils.initialized():
        #     self.loss_fn[loss] = DDPLoss(self.loss_fn[loss])
    return loss_fn