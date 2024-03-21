"""
DEQ-INR maps the input grid $(x, y)$ to an RGB value $(r,g,b)$ based on the fixed points of the periodical MLP layer.
"""


import numpy as np

import torch
import torch.nn as nn

from torchdeq import get_deq
from torchdeq.norm import apply_norm, reset_norm
from torchdeq.loss import fp_correction


import skimage
import matplotlib.pyplot as plt
from tqdm import tqdm


######################################
# Linear Layers
######################################


class INRInjection(nn.Linear):
    def __init__(self, in_features, d_hidden, scale=256):
        super().__init__(in_features, d_hidden)
        self.weight.data *= scale
        self.bias.data.uniform_(-np.pi, np.pi)


class MFNLinear(nn.Linear):
    def __init__(self, d_hidden, scale=256):
        super().__init__(d_hidden, d_hidden)
        nn.init.uniform_(self.weight, -np.sqrt(1 / d_hidden), np.sqrt(1 / d_hidden))


######################################
# DEQ-INR Base class
######################################


class DEQSirenINR(nn.Module):
    def __init__(
        self,
        d_in=2,
        d_out=3,
        d_hidden=128,
        n_layer=1,
        scale=256.0,
        deq_mode=True,
        **kwargs,
    ):
        super().__init__()

        # encoder
        self.inj = INRInjection(d_in, d_hidden * (n_layer + 1), scale=scale)
        # implicit layer
        self.mfn = nn.ModuleList(
            [MFNLinear(d_hidden, d_hidden) for _ in range(n_layer)]
        )
        # decoder
        self.out = nn.Linear(d_hidden, d_out)

        self.d_hidden = d_hidden
        self.n_layer = n_layer

        self.register_buffer("z_aux", self._init_z())
        self.deq_mode = deq_mode
        self.deq = get_deq(**kwargs)

        # This function automatically decorates weights in your DEQ layer
        # to have weight/spectral normalization. (for better stability)
        # Using norm_type='none' in `kwargs` can also skip it.
        apply_norm(self.mfn, **kwargs)

    def _init_z(self):
        return torch.zeros(1, self.d_hidden)

    def injection(self, x):
        u = self.inj(x)
        u = u.chunk(self.n_layer + 1, dim=1)
        return u

    def filter(self, z, u):
        """Siren-alike additive filter"""
        return torch.sin(z + u)

    def mfn_forward(self, z, u):
        """
        Multiplicative filter networks
        """
        # Fixed point reuse
        z = z + self.filter(self.z_aux, u[0])

        for i, layer in enumerate(self.mfn):
            z = self.filter(layer(z), u[i + 1])

        return z

    def forward(self, x, z=None):
        reuse = True
        if z is None:
            z = torch.zeros(x.shape[0], self.d_hidden).to(x)
            reuse = False

        # encode
        u = self.injection(x)

        reset_norm(self.mfn)
        f = lambda z: self.mfn_forward(z, u)
        if self.deq_mode:
            solver_kwargs = {"f_max_iter": 0} if reuse else {}
            z_pred, info = self.deq(f, z, solver_kwargs=solver_kwargs)
        else:
            z_pred = [f(z)]

        # decode
        outputs = [self.out(z) for z in z_pred]

        return outputs, z_pred[-1]


def test_model():
    x = torch.rand(512 * 512, 2).cuda()
    model = DEQSirenINR(
        d_in=2, d_out=3, d_hidden=256, n_layer=1, scale=256.0, deq_mode=True, ift=True
    ).cuda()
    out, z = model(x)


######################################
# Utils
######################################


def visualize_results(y_orig, y_pred, grid_size=128):
    y_orig = postprocess(y_orig).detach().cpu().numpy()
    y_pred = postprocess(y_pred).detach().cpu().numpy()
    data_channels = y_orig.shape[-1]

    fig, ax = plt.subplots(1, 2, figsize=(10, 24))
    if data_channels == 1:
        ax[0].imshow(y_orig.reshape(grid_size, grid_size), cmap="gray")
        ax[1].imshow(y_pred.reshape(grid_size, grid_size), cmap="gray")
    else:
        ax[0].imshow(y_orig.reshape((grid_size, grid_size, data_channels)))
        ax[1].imshow(y_pred.reshape((grid_size, grid_size, data_channels)))

    ax[0].set_title("GT")
    ax[0].axis("off")
    ax[1].set_title("Reconstruction")
    ax[1].axis("off")


def plot_training_stats(loss_log, psnr_log):
    fig, ax1 = plt.subplots()
    ax1.plot(loss_log, color="orange", label="Train Loss")
    ax1.set_xlabel("epoch")
    ax1.semilogy()

    ax2 = ax1.twinx()
    ax2.plot(psnr_log, color="b", label="PSNR")
    plt.title("training curve")

    ax1.legend(loc="best")
    ax2.legend(loc="best")


# Normalize input np.array
# Convert np.array to tensor
def preprocess(x, normalize=False, device=None, dtype=None):
    if normalize:
        x = 2 * x - 1
    B, H, W, C = x.shape

    x = torch.tensor(x, device=device).float()
    return x.reshape(B * H * W, C)


# Convert tensor to np.array
# Clip min/max values
def postprocess(x):
    return torch.clip((x.detach() + 1) / 2, 0, 1)


# Calculate PSNR
def PSNR(y_orig, y_pred):
    mse = torch.mean((y_orig - y_pred) ** 2)
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()


def load_data(img, grid_size, device=None, show_img=True):
    # Load image
    if img == "astronaut":
        image = skimage.data.astronaut()
        dataset = image.reshape(1, 512, 512, 3).astype(np.float32) / 255
    else:
        image = skimage.data.camera()
        dataset = image.reshape(1, 512, 512, 1).astype(np.float32) / 255

    data_channels = dataset.shape[-1]
    RES = image.shape[1]

    full_x = np.linspace(0, 1, RES) * 2 - 1
    full_x_grid = np.stack(np.meshgrid(full_x, full_x), axis=-1)[None, :, :]

    # Downsampling
    x_train_data = x_test_data = full_x_grid[
        :, :: RES // grid_size, :: RES // grid_size
    ]
    y_train_data = y_test_data = dataset[:, :: RES // grid_size, :: RES // grid_size]
    print("Shape of x_train", x_train_data.shape)
    print("Shape of y_train", y_train_data.shape)

    # Visualize the image if requested
    if show_img:
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        if img == "astronaut":
            ax[0].imshow(y_train_data[0, :, :, :])
        else:
            ax[0].imshow(y_train_data[0, :, :, 0], cmap="gray")
            ax[0].set_title("image")
            ax[0].axis("off")
            ax[1].imshow(x_train_data[0, :, :, 0])
            ax[1].set_title("horizontal mesh_grid")
            ax[1].axis("off")
            ax[2].imshow(x_train_data[0, :, :, 1])
            ax[2].set_title("vertical mesh_grid")
            ax[2].axis("off")

    # Transform to tensors
    x_train = preprocess(x_train_data, device=device)
    y_train = preprocess(y_train_data, normalize=True, device=device)

    return x_train, y_train


######################################
# Training example
######################################


def train(
    model, x_train, y_train, lr=1e-3, log_freq=100, epochs=10, deq_mode=True, reuse=True
):
    # Compute compression rate
    n_params = sum(p.numel() for p in model.parameters())
    img_size = y_train.numel()
    print(f"Model Parameters: {n_params:,}")
    print(f"Img Size: {img_size:,}")
    print(f"Compression ratio: {n_params/img_size:.3f}")

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=1e-6
    )

    loss_log, psnr_log = [], []

    z_star = None
    for i in tqdm(range(epochs)):
        y_pred, z_pred = model(x_train, z_star)
        if deq_mode and reuse:
            z_star = z_pred.detach()

        # If you use trajectory sampling, fp_correction automatically
        # aligns the tensors and applies your loss function.
        loss_fn = lambda y_gt, y: ((y_gt - y) ** 2).mean()
        train_loss = fp_correction(loss_fn, (y_train, y_pred))

        train_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_log.append(train_loss.item())

        train_psnr = PSNR(postprocess(y_train), postprocess(y_pred[-1]))
        psnr_log.append(train_psnr)

        # Log
        if (i + 1) % log_freq == 0:
            with torch.inference_mode():
                y_pred, _ = model(x_train)
                psnr = PSNR(postprocess(y_train), postprocess(y_pred[-1]))
                print(
                    f"[Train] Loss: {train_loss.item():.5f}, || PSNR: {psnr:.2f} || Lr: {scheduler.get_last_lr()}"
                )

        optimizer.zero_grad()

    return y_pred[-1], loss_log, psnr_log


if __name__ == "__main__":
    # Configs
    grid_size = 256
    img = "astronaut"

    lr = 1e-2
    epochs = 500

    filter = "Fourier"
    d_hidden = 256
    scale = 256.0

    deq_mode = False
    reuse = False
    ift = False
    grad = 3
    norm_type = "none"

    device = torch.device("cuda:0")
    log_freq = 100

    x_train, y_train = load_data(img, grid_size, device=device)
    model = DEQSirenINR(
        d_in=2,
        d_out=y_train.shape[-1],
        d_hidden=d_hidden,
        n_layer=1,
        scale=scale,
        deq_mode=deq_mode,
        ift=ift,
        grad=grad,
        norm_type=norm_type,
    ).to(device)

    exp_y_pred, loss_log, psnr_log = train(
        model, x_train, y_train, lr=lr, log_freq=log_freq
    )

    plot_training_stats(loss_log, psnr_log)
    visualize_results(y_train, exp_y_pred, grid_size=grid_size)
