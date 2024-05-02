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

# 
# https://omoindrot.github.io/triplet-loss
def _pairwise_distances(embeddings, squared=False, save_squared=False):
    """Compute the 2D matrix of distances between all the embeddings.

    Args:
        embeddings: tensor of shape (batch_size, embed_dim)
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        pairwise_distances: tensor of shape (batch_size, batch_size)
    """
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    # When input is a 2-D tensor torch.t is equivalent to transpose(input, 0, 1).
    dot_product = torch.matmul(embeddings, torch.t(embeddings))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute the pairwise distance matrix as we have:
    # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    # shape (batch_size, batch_size)
    # distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = torch.unsqueeze(square_norm, 0) - 2.0 * dot_product + torch.unsqueeze(square_norm, 1)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.clamp(distances, min=0.0)

    if not squared:
        if save_squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = torch.where(distances == 0.0, 1, 0).float()
            distances = distances + mask * 1e-16

            distances = torch.sqrt(distances)

            # Correct the epsilon added: set the distances on the mask to be exactly 0.0
            distances = distances * (1.0 - mask)
        else:
            distances = torch.sqrt(distances)

    return distances

def contrastive_loss(fixedpoints, data, closs_type="next", squared=True):
    """ Fixed-point contrastive loss.
    fixedpoints = info["z_pred"][-1]
    """
    # Fixed-point contrastive loss
    # fixedpoints E1: [batch_size*num_atoms, irrep_dim]
    # fixedpoints E2: [batch_size*num_atoms, num_coefficients, num_channels]
    # reshape to [batch_size, num_atoms, ...]
    dims_per_atom = fixedpoints.shape[1:]
    # natoms: number atoms per batch [batch_size]
    batch_size = len(data.natoms)
    num_atoms = data.natoms[0]

    # V1 view / reshape
    # data.batch contains the batch index for each atom (node)
    fixedpoints = fixedpoints.view(batch_size, num_atoms, *dims_per_atom)

    # reshape to [batch_size, features]
    fixedpoints = fixedpoints.reshape(batch_size, -1)

    # compute pairwise distances between fixed points
    # [batch_size, batch_size]
    # similarity = torch.cdist(fp_reshaped, fp_reshaped, p=2) # BxPxM, BxRxM -> BxPxR
    distances = _pairwise_distances(fixedpoints, squared=squared)

    # Similarity matrix 
    # for the contrastive loss we construct a matrix of positive and negative relations
    # similarity = torch.zeros(batch_size, batch_size, device=fixedpoints.device, dtype=fixedpoints.dtype)
    
    if closs_type == "next":
        # |FP(x_t) - FP(x_t+/-1)| -> diagonal shifted right-up by 1
        similarity = torch.diag(
            torch.ones(batch_size-1, device=fixedpoints.device, dtype=fixedpoints.dtype), diagonal=1
        )
    else:
        raise NotImplementedError(f"Unknown closs_type: {closs_type}")
    
    return (distances * similarity).mean()