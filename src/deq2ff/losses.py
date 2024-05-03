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

class TripletLoss(torch.nn.Module):
    # https://medium.com/@Skpd/triplet-loss-on-imagenet-dataset-a2b29b8c2952
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        dim = [i for i in range(1, len(x1.shape))]
        return (x1 - x2).pow(2).sum(dim=dim)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

def calc_triplet_loss(fixedpoints, data, triplet_lossfn):
    """
    Assumes:
    - fixedpoints: [batch_size*num_atoms, ...]
    - positive/negative pairs and buffers are ordered: 
        [p1 p1 p2 p2 (b1) (b1) (b2) (b2) ... n1 n2]
    """
    # Fixed-point contrastive loss
    # fixedpoints E1: [batch_size*num_atoms, irrep_dim]
    # fixedpoints E2: [batch_size*num_atoms, num_coefficients, num_channels]
    dims_per_atom = fixedpoints.shape[1:]
    # natoms: number atoms per batch [batch_size]
    batch_size = len(data.natoms)
    num_atoms = data.natoms[0]

    # reshape to [batch_size, num_atoms, ...]
    # data.batch contains the batch index for each atom (node)
    fixedpoints = fixedpoints.view(batch_size, num_atoms, *dims_per_atom)

    # reshape to [batch_size, features]
    # fixedpoints = fixedpoints.reshape(batch_size, -1)

    triplets_per_batch = batch_size // 3
    anchors = fixedpoints[:triplets_per_batch]
    positives = fixedpoints[triplets_per_batch:2*triplets_per_batch]
    negatives = fixedpoints[batch_size-triplets_per_batch:]

    return triplet_lossfn(anchors, positives, negatives)


class TripletDataloader:
    def __init__(self, dataset, batch_size, drop_last=True, random_start=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_batches = len(dataset) // batch_size

        # Idea: split up dataset into two unequal parts
        # first part: 2/3 + buffer of the dataset for the positive pairs 
        # second part: 1/3 of the dataset for the negative pairs
        # a batch will be formed by taking 
        # n*2 + n*buffer_per_batch samples from the first part and n*1 samples from the second part
        # where n = triplets_per_batch
        # batch_size = 8: [p1 p1 p2 p2 b1 b2 ... n1 n2] (positive, buffer, negative)

        self.triplets_per_batch = batch_size // 3
        self.buffer_per_batch = batch_size % 3
        # size of the first / second part of the batch
        self.bs1 = 2 * self.triplets_per_batch + self.buffer_per_batch
        self.bs2 = self.triplets_per_batch
        # size of the first part of the dataset = size of jump from p1 to n1
        self.ds1 = self.bs1 * self.num_batches

        # start at random position in the dataset
        self.random_start = random_start

        self.start = 0
        # current index in the dataset where the next batch starts
        self.idx = 0 
        # current batch index
        self.ibatch = 0
        self.reset()

    def reset(self, random_start=None):
        # start at random position in the dataset
        if random_start is None:
            random_start = self.random_start
        else:
            self.random_start = random_start
        if random_start:
            self.start = torch.randint(0, len(self.dataset), (1,)).item()
        else:
            self.start = 0
        # reset idx to start
        self.idx = self.start
        self.ibatch = 0

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.ibatch >= self.num_batches:
            raise StopIteration
        
        # prefer not to use random_start with indices tensors here.
        # indexing a tensor with slice or int returns a view of that without copying its underlying storage 
        # but indexing with another tensor (a Bool or a Long one but not a 0-dim long tensor) or a list returns a copy of the tensor
        # (You can use .storage().data_ptr() to see if the underlying data of a tensor has been copied or not.)
        if self.random_start:
            # indices for the current batch 
            indices1 = torch.arange(self.idx, self.idx+self.bs1)
            indices2 = torch.arange(self.idx+self.ds1, self.idx+self.ds1+self.bs2)
            # if we start from a random position, we need to wrap around the dataset
            indices1 = torch.where(indices1 >= len(self.dataset), indices1 - len(self.dataset), indices1).long()
            indices2 = torch.where(indices2 >= len(self.dataset), indices2 - len(self.dataset), indices2).long()
            # using indices tensor will copy the data
            # print("dataset storage                             ", self.dataset.storage().data_ptr())
            # print("dataset[self.idx, self.idx+self.bs1] storage", self.dataset[self.idx:self.idx+self.bs1].storage().data_ptr())
            # print("dataset[indices1] storage                   ", self.dataset[indices1].storage().data_ptr())
            # build the batch
            batch = torch.concat([
                self.dataset[indices1],
                self.dataset[indices2]
            ], dim=0)

        else:
            batch = torch.concat([
                self.dataset[self.idx : self.idx+self.bs1],
                self.dataset[self.idx+self.ds1 : self.idx+self.ds1+self.bs2]
            ], dim=0)

        # update counters
        self.idx += self.bs1
        self.ibatch += 1
        return batch