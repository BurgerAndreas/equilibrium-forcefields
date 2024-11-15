import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import umap

import matplotlib.pyplot as plt
import seaborn as sns
from deq2ff.plotting.style import (
    chemical_symbols,
    plotfolder,
    set_seaborn_style,
    reset_plot_styles,
    set_style_after,
    PALETTE,
    cdict,
)

from e3nn import o3
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from deq2ff.plotting.style import chemical_symbols, plotfolder, set_seaborn_style, set_style_after

from deq2ff.logging_utils import init_wandb, fix_args_set_name
import scripts as scripts
from scripts.train_deq_md import train_md, equivariance_test

# register all models
import deq2ff.register_all_models


def get_pairwise_distances(tensors):
    """Pass a list of tensors, get list of distances between all pairs"""
    # Naive version
    distances = []
    for i, ti in enumerate(tensors):
        for j in range(i, len(tensors)):
            distances.append(torch.linalg.norm(ti - tensors[j]))
    # faster version using matrix math?
    return distances


def compute_mae(pred_y, pred_dy, data, normalizers, criterion_energy, criterion_force):
    
    target_y = normalizers["energy"](data.y, data.z)  # [NB], [NB]
    target_dy = normalizers["force"](data.dy, data.z)
    
    # reshape model output [B] (OC20) -> [B,1] (MD17)
    if pred_y.dim() == 1:
        pred_y = pred_y.unsqueeze(-1)

    # reshape data [B,1] (MD17) -> [B] (OC20)
    # if squeeze_e_dim and target_y.dim() == 2:
    #     target_y = target_y.squeeze(1)

    loss_e = criterion_energy(pred_y, target_y)
    loss_f = criterion_force(pred_dy, target_dy)

    return loss_e, loss_f

"""
Usage
Get checkpoint from cluster
srcdir=/home/andreasburger/equilibrium-forcefields
trgtdir=.
clusterlogin=andreasburger@tacozoid.accelerationconsortium.ai
checkpointpath=/models/md17/deq_equiformer_v2_oc20/aspirin/pDEQsap2/final_epochs@499_e@1.7924_f@0.4007.pth.tar
scp ${clusterlogin}:${srcdir}/${checkpointpath} ${trgtdir}/${checkpointpath}
"""


@hydra.main(
    config_name="md17", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    #############################
    # config
    args.batch_size = 1
    args.fpreuse_test = True

    total_samples = 1000
    consecutive = True

    if consecutive:
        # Consecutive samples
        args.test_patch_size = total_samples
        args.test_patches = 1
    else:
        # Non-consecutive samples
        # needs to be an even number, s.t. we can use fpreuse every second datadpoint
        args.test_patch_size = 2
        # the more the longer. 10 to 10000. Default: 1000
        args.test_patches = total_samples // args.test_patch_size


    #############################
    # pDEQsapt

    # get data
    args.return_model_and_data = True
    # ensure we load a checkpoint of a trained model
    args.assert_checkpoint = 0.1

    # init_wandb(args, project="equilibrium-forcefields-equiformer_v2")
    args.wandb = False  # TODO: we are not logging anything
    
    args = fix_args_set_name(args)

    # to dict
    args = OmegaConf.structured(OmegaConf.to_yaml(args))

    datas = train_md(args)
    model = datas["model"]
    # train_dataset = datas["train_dataset"]
    # test_dataset_full = datas["test_dataset_full"]
    # test_dataset = datas["test_dataset"]
    test_loader = datas["test_loader"]
    normalizers = datas["normalizers"]
    criterion_energy = datas["criterion_energy"]
    criterion_force = datas["criterion_force"]
    device = list(model.parameters())[0].device
    dtype = model.parameters().__next__().dtype

    # eval mode
    model.eval()
    
    def diff_vector(x,y): 
        """[N,3] -> [1]"""
        assert x.dim() == 2 and y.dim() == 2, f"{x.shape=}, {y.shape=}"
        return torch.mean(torch.linalg.norm(x-y, dim=-1))
    
    def diff_scalar(x,y): 
        """[1] -> [1]"""
        assert x.dim() in [0,1] and y.dim() in [0,1], f"{x.shape=}, {y.shape=}"
        return torch.abs(x-y)
    
    def diffrel_vector(x, y): 
        """[N,3] -> [1]"""
        assert x.dim() == 2 and y.dim() == 2, f"{x.shape=}, {y.shape=}"
        return torch.mean( # [N] -> [1]
            torch.linalg.norm(x-y, dim=-1) # [N,3] -L2> [N]
            / torch.mean( # [N,2] -> [N]
                torch.stack([ # [N],[N] -> [N,2]
                    torch.linalg.norm(x, dim=-1), # [N,3] -L2> [N]
                    torch.linalg.norm(y, dim=-1) # [N,3] -L2> [N]
                ], dim=-1), 
                dim=-1
            )
        )
    
    def diffrel_scalar(x, y): 
        """[1] -> [1]"""
        assert x.dim() in [0,1] and y.dim() in [0,1], f"{x.shape=}, {y.shape=}"
        return torch.abs(x-y) / torch.mean(torch.stack([x,y])) # [1] -> [1]

    # collate = Collater(follow_batch=None, exclude_keys=None)

    with torch.no_grad():

        # temp variables
        forces = []
        forces_wreuse = []
        fmae = []
        fmae_wreuse = []
        energies = []
        energies_wreuse = []
        emae = []
        emae_wreuse = []
        
        # difference w/wo reuse
        d_forces = []
        d_energies = []
        d_f_rel = []
        d_e_rel = []
        d_fmae = []
        d_fmae_rel = []
        d_emae = []
        d_emae_rel = []
        

        # loop over test_dataset
        fixedpoint = None
        print("-"*80)
        for cnt, data in enumerate(tqdm(test_loader, desc="Predicting")):
            data = data.to(device)
            data = data.to(device, dtype)

            if cnt >= total_samples:
                break
            
            # first prediction without fixed-point reuse
            pred_y1, pred_dy1, info = model(
                data=data,  # for EquiformerV2
                node_atom=data.z,
                pos=data.pos,
                batch=data.batch,
                # step=pass_step,
                # datasplit=_datasplit,
                return_fixedpoint=False,
                # fixedpoint=fixedpoint,
                # solver_kwargs=solver_kwargs,
            )
            pred_y1, pred_dy1 = pred_y1.detach(), pred_dy1.detach()
            forces.append(pred_dy1)
            energies.append(pred_y1)
            fmae1, emae1 = compute_mae(pred_y1, pred_dy1, data, normalizers, criterion_energy, criterion_force)
            fmae.append(fmae1)
            emae.append(emae1)

            # second prediction with fixed-point reuse
            pred_y2, pred_dy2, fp, info = model(
                data=data,  # for EquiformerV2
                node_atom=data.z,
                pos=data.pos,
                batch=data.batch,
                # step=pass_step,
                # datasplit=_datasplit,
                return_fixedpoint=True,
                fixedpoint=fixedpoint,
                # solver_kwargs=solver_kwargs,
            )
            pred_y2, pred_dy2 = pred_y2.detach(), pred_dy2.detach()
            fixedpoint = fp.detach()
            forces_wreuse.append(pred_dy2)
            energies_wreuse.append(pred_y2)
            fmae2, emae2 = compute_mae(pred_y2, pred_dy2, data, normalizers, criterion_energy, criterion_force)
            fmae_wreuse.append(fmae2)
            emae_wreuse.append(emae2)

            # compute difference
            # dy=force [N,3], y=energy [1]
            d_forces.append(diff_vector(pred_dy2, pred_dy1))
            d_energies.append(diff_scalar(pred_y2, pred_y1))
            # relative
            d_f_rel.append(diffrel_vector(pred_dy2, pred_dy1))
            d_e_rel.append(diffrel_scalar(pred_y2, pred_y1))
            # MAE: [1]
            d_fmae.append(diff_scalar(fmae2, fmae1))
            d_emae.append(diff_scalar(emae2, emae1))
            # relative
            d_fmae_rel.append(diffrel_scalar(fmae2, fmae1))
            d_emae_rel.append(diffrel_scalar(emae2, emae1))

        # no_grad ends

    # Summary statistics
    forces = torch.stack(forces)
    forces_wreuse = torch.stack(forces_wreuse)
    fmae = torch.stack(fmae)
    fmae_wreuse = torch.stack(fmae_wreuse)
    energies = torch.stack(energies)
    energies_wreuse = torch.stack(energies_wreuse)
    emae = torch.stack(emae)
    emae_wreuse = torch.stack(emae_wreuse)
    d_forces = torch.stack(d_forces)
    d_energies = torch.stack(d_energies)
    d_fmae = torch.stack(d_fmae)
    d_emae = torch.stack(d_emae)
    d_f_rel = torch.stack(d_f_rel)
    d_e_rel = torch.stack(d_e_rel)
    d_fmae_rel = torch.stack(d_fmae_rel)
    d_emae_rel = torch.stack(d_emae_rel)
    
    for m in [d_forces, d_energies, d_fmae, d_emae, d_f_rel, d_e_rel, d_fmae_rel, d_emae_rel]:
        print(f"{m.shape=}")
    
    print("-"*80)
    
    print(" ")
    print(f"avg force difference:  {d_forces.mean().item():.3f} +/- {d_forces.std().item():.3f}")
    print(f"avg energy difference: {d_energies.mean().item():.3f} +/- {d_energies.std().item():.3f}")
    
    print(" ")
    print(f"avg force rel diff:    {d_f_rel.mean().item():.3f} +/- {d_f_rel.std().item():.3f}")
    print(f"avg energy rel diff:   {d_e_rel.mean().item():.3f} +/- {d_e_rel.std().item():.3f}")
    
    print(" ")
    print(f"avg fmae difference:   {d_fmae.mean().item():.3f} +/- {d_fmae.std().item():.3f}")
    print(f"avg emae difference:   {d_emae.mean().item():.3f} +/- {d_emae.std().item():.3f}")
    
    print(" ")
    print(f"avg fmae rel diff:     {d_fmae_rel.mean().item():.3f} +/- {d_fmae_rel.std().item():.3f}")
    print(f"avg emae rel diff:     {d_emae_rel.mean().item():.3f} +/- {d_emae_rel.std().item():.3f}")

    print(" ")
    print(f"max force difference:  {d_forces.max().item():.3f}")
    print(f"max energy difference: {d_energies.max().item():.3f}")
    print(f"max fmae difference:   {d_fmae.max().item():.3f}")
    print(f"max emae difference:   {d_emae.max().item():.3f}")
    print(f"max force rel diff:    {d_f_rel.max().item():.3f}")
    print(f"max energy rel diff:   {d_e_rel.max().item():.3f}")
    
    
    print(" ")
    set_seaborn_style()
    
    # remove first sample
    d_forces = d_forces[1:]
    d_f_rel = d_f_rel[1:]
    
    # remove largest sample
    d_f_rel = d_f_rel[d_f_rel != d_f_rel.max()]
    
    # plot force difference over time
    fig = plt.figure()
    plt.plot(d_forces.cpu().numpy())
    plt.title(f"Force difference w/wo FP reuse")
    plt.ylabel("Force difference")
    plt.xlabel("Sample index")
    set_style_after(ax=plt.gca(), legend=False)
    fname = f"force_diff.png"
    plt.savefig(fname)
    print(f"Saved to {fname}")
    plt.close()
    
    # plot relative force difference over time 
    fig = plt.figure()
    plt.plot(d_f_rel.cpu().numpy())
    plt.title(f"Relative force difference w/wo FP reuse")
    plt.ylabel("Relative force difference")
    plt.xlabel("Sample index")
    set_style_after(ax=plt.gca(), legend=False)
    fname = f"force_rel_diff.png"
    plt.savefig(fname)
    print(f"Saved to {fname}")
    plt.close()
    
    print("\nDone!")


if __name__ == "__main__":
    hydra_wrapper()
