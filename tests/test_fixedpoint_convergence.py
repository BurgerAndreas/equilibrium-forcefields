import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb
import os

from e3nn import o3
from torch_geometric.data import Data
from torch_geometric.loader.dataloader import Collater

from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from deq2ff.plotting.style import chemical_symbols, plotfolder

from deq2ff.logging_utils import init_wandb
import scripts as scripts
from scripts.train_deq_md import train, equivariance_test

# register all models
import deq2ff.register_all_models

def get_pairwise_distances(tensors):
    """Pass a list of tensors, get list of distances between all pairs"""
    # Naive version
    distances = []
    for i, ti in enumerate(tensors):
        for j in range(i, len(tensors)):
            distances.append(
                torch.linalg.norm(ti - tensors[j])
            )
    # faster version using matrix math?
    return distances

@hydra.main(
    config_name="md17", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    # config
    args.batch_size = 1
    args.fpreuse_test = True

    args.test_patch_size = 2 # needs to be an even number, s.t. we can use fpreuse every second datadpoint
    args.test_patches = 10 # the more the longer. 10 to 10000. Default: 1000

    # get data
    args.return_model_and_data = True
    # ensure we load a checkpoint of a trained model
    # args.assert_checkpoint = True

    # init_wandb(args, project="equilibrium-forcefields-equiformer_v2")
    args.wandb = False # TODO: we are not logging anything
    run_id = init_wandb(args)

    datas = train(args)
    model = datas["model"]
    # train_dataset = datas["train_dataset"]
    # test_dataset_full = datas["test_dataset_full"]
    test_dataset = datas["test_dataset"]
    
    device = list(model.parameters())[0].device
    dtype = model.parameters().__next__().dtype

    # collate = Collater(follow_batch=None, exclude_keys=None)

    with torch.no_grad():
        
        # temp variables
        fp1 = None
        fp2 = None
        fp2_wreuse = None
        # save for statistics
        d_fp2_wo_reuse = []
        d_fp2_fp1 = [] # fp2 - fp1
        d_fp2wreuse_fp1 = [] # fp2_wreuse - fp1
        # save for statistics about repeats
        fp_repeat_avg_distances = []
        fp_repeat_max_distances = []
        
        # loop over test_dataset
        for cnt, data in enumerate(test_dataset):
            data = data.to(device)
            data = data.to(device, dtype)

            fpreuse = True if (cnt % 0 == 0) else False

            # TODO: compute fixed-point multiple times to check 
            # if we converge to the same fixed-point / the fixed-point is unique
            max_repeats = 1
            # max_repeats = 10 if cnt == 0 else max_repeats
            reps_fp1 = []
            reps_fp2 = []
            reps_fp2_wreuse = []

            for rep in range(max_repeats):

                if fpreuse is False:
                    # compute first fixed-point 
                    pred_y, pred_dy, fp1, info = model(
                        data=data,  # for EquiformerV2
                        node_atom=data.z,
                        pos=data.pos,
                        batch=data.batch,
                        # step=pass_step,
                        # datasplit=_datasplit,
                        return_fixedpoint=True,
                        # fixedpoint=fixedpoint,
                        # solver_kwargs=solver_kwargs,
                    )
                    # reps.append(copy.deepcopy(fp1))
                    reps_fp1.append(fp1)

                else:
                    # compute second fixed-point WITHOUT reuse
                    pred_y, pred_dy, fp2, info = model(
                        data=data,  # for EquiformerV2
                        node_atom=data.z,
                        pos=data.pos,
                        batch=data.batch,
                        # step=pass_step,
                        # datasplit=_datasplit,
                        return_fixedpoint=True,
                        # fixedpoint=fixedpoint,
                        # solver_kwargs=solver_kwargs,
                    )
                    reps_fp2.append(fp2)

                    # compute second fixed-point WITH reuse
                    pred_y, pred_dy, fp2_wreuse, info = model(
                        data=data,  # for EquiformerV2
                        node_atom=data.z,
                        pos=data.pos,
                        batch=data.batch,
                        # step=pass_step,
                        # datasplit=_datasplit,
                        return_fixedpoint=True,
                        fixedpoint=fp1,
                        # solver_kwargs=solver_kwargs,
                    )
                    reps_fp2_wreuse.append(fp2_wreuse)

                    # compute distances once per data point
                    if rep == 0:
                        # compute distance between fixed-points w/o reuse
                        d_fp2_wo_reuse.append(torch.linalg.norm(fp2 - fp2_wreuse))
                        # comparison
                        d_fp2_fp1.append(torch.linalg.norm(fp2 - fp1))
                        d_fp2wreuse_fp1.append(torch.linalg.norm(fp2_wreuse - fp1))
                
                # reps finished
                
                # print statistics once we finished multiple repeats
                if max_repeats > 1 and rep == max_repeats - 1:
                    for reps, _name in zip([reps_fp1, reps_fp2, reps_fp2_wreuse], ["fp1", "fp2", "fp2_wreuse"]):
                        if len(reps) > 1:
                            pass
                            # compute the pairwise distances between the fixed-points in the list
                            pairwise_distances = get_pairwise_distances(reps)
                            # print the average distance

                            fp_repeat_avg_distances.append()
                            # print the max distance

                            fp_repeat_max_distances.append()
            
            # test data points finished
        
        # no_grad ends
    
    print("\nDo we always converge to the same fixed-points?")
    print("Average distance between fixed-points, averaged over data points:", torch.mean(fp_repeat_avg_distances).item())
    print("Max     distance between fixed-points, averaged over data points:", torch.mean(fp_repeat_max_distances).item())
        
    # aggregate statistics
    print("\nAre fixed-points w/o reuse close? (i.e. does fixed-point reuse lead to different fixed-points?)")
    print("|fp2 - fp2_wreuse| =", torch.mean(d_fp2_wo_reuse).item())
    print("|fp2 - fp1|        =", torch.mean(d_fp2_wo_reuse).item(), "<- should be lower")
    print("|fp2_wreuse - fp1| =", torch.mean(d_fp2_wo_reuse).item(), "<- should be lower")


    # plot the full distributions
    # df = 

    # import seaborn as sns

    

    print('\nDone!')


if __name__ == "__main__":
    hydra_wrapper()