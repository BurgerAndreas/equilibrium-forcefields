import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import wandb
import os
import umap

import matplotlib.pyplot as plt
import seaborn as sns
from deq2ff.plotting.style import chemical_symbols, plotfolder, set_seaborn_style, reset_plot_styles, set_style_after, PALETTE, cdict

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


@hydra.main(
    config_name="md17", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    #############################
    # config
    args.batch_size = 1
    args.fpreuse_test = True
    
    total_samples = 200
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


    # TODO: compute fixed-point multiple times to check
    # if we converge to the same fixed-point / the fixed-point is unique
    max_repeats = 1 # 1, 10

    repeat_tol = 1e-5

    # If false, consider each node in the batch as a separate datapoint
    # for the UMAP plot
    batch_as_one = True
    do_umap = True

    do_pca = False
    
    samples_for_projection = 100
    assert samples_for_projection * 2 >= args.test_patch_size * args.test_patches, \
        f"Need more samples for projection"
    do_proj = do_pca or do_umap
    if do_proj:
        assert max_repeats == 1, "Cannot project multiple repetitions of same data point"

    #############################
    # pDEQsapt

    # get data
    args.return_model_and_data = True
    # ensure we load a checkpoint of a trained model
    args.assert_checkpoint = 0.1

    # init_wandb(args, project="equilibrium-forcefields-equiformer_v2")
    args.wandb = False  # TODO: we are not logging anything
    run_id = init_wandb(args)

    datas = train_md(args)
    model = datas["model"]
    # train_dataset = datas["train_dataset"]
    # test_dataset_full = datas["test_dataset_full"]
    # test_dataset = datas["test_dataset"]
    test_loader = datas["test_loader"]

    device = list(model.parameters())[0].device
    dtype = model.parameters().__next__().dtype

    # eval mode
    model.eval()

    # collate = Collater(follow_batch=None, exclude_keys=None)

    max_batches = len(test_loader)

    with torch.no_grad():

        # temp variables
        fp1 = None
        fp2 = None
        fp2_wreuse = None
        # save for umap
        fp1s = []
        fp2s = []
        fp2_wreuses = []
        # save for statistics
        d_fp2_wo_reuse = []
        d_fp2_wo_reuse_rel = []
        d_fp2_fp1 = []  # fp2 - fp1
        d_fp2wreuse_fp1 = []  # fp2_wreuse - fp1
        # save for statistics about repeats
        fp_repeat_avg_distances = []
        fp_repeat_max_distances = []
        fp_repeat_over_tol = []

        # loop over test_dataset
        for cnt, data in enumerate(test_loader):
            data = data.to(device)
            data = data.to(device, dtype)

            fpreuse = True if (cnt % 2 == 1) else False

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
                    fp1s.append(fp1)

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
                    fp2s.append(fp2)

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
                    fp2_wreuses.append(fp2_wreuse)

                    # compute distances once per data point
                    if (rep == 0) and (fpreuse is True):
                        # compute distance between fixed-points w/o reuse
                        d_fp2_wo_reuse.append(torch.linalg.norm(fp2 - fp2_wreuse))
                        # comparison
                        d_fp2_fp1.append(torch.linalg.norm(fp2 - fp1))
                        d_fp2wreuse_fp1.append(torch.linalg.norm(fp2_wreuse - fp1))
                        print(
                            f"\n{cnt}: |fp2 - fp2_wreuse| =", d_fp2_wo_reuse[-1].item()
                        )
                        # print(f"{cnt}: |fp2 - fp1|        =", d_fp2_fp1[-1].item(), "<- should be larger")
                        # print(f"{cnt}: |fp2_wreuse - fp1| =", d_fp2wreuse_fp1[-1].item(), "<- should be larger")
                        print(
                            f"{cnt}: |fp2 - fp2_wreuse| / |fp2 - fp1| =",
                            d_fp2_wo_reuse[-1].item() / d_fp2_fp1[-1].item(),
                            "<- should be small",
                        )
                        _rel_err = (d_fp2_wo_reuse[-1] / torch.linalg.norm(fp2))
                        print(
                            f"{cnt}: |fp2 - fp2_wreuse| / |fp2| =",
                            _rel_err.item(),
                            "<- should be small",
                        )
                        d_fp2_wo_reuse_rel.append(_rel_err)
                        # TODO: compare the distances between the predicted forces

                    # plot distances once per run
                    print(f"do_umap={do_umap}, rep={rep}, fpreuse={fpreuse}, cnt={cnt}")
                    if do_umap and (rep == 0) and (fpreuse is True) and (cnt + 1 >= samples_for_projection * 2):
                        print("")
                        print("-" * 60)
                        print("Plotting UMAP")
                        # UMAP neeeds shape (NumSamples, Rest)
                        # Tensors are shape (B, H, C) 
                        flatpoints = []
                        for _fp1, _fp2, _fp2_wreuse in zip(fp1s, fp2s, fp2_wreuses):
                            if batch_as_one:
                                # Flatten (B, H, C) to (1, B * H * C)
                                fp1_flat = _fp1.view(1, -1)  # (1, B * H * C)
                                fp2_flat = _fp2.view(1, -1)
                                fp2wr_flat = _fp2_wreuse.view(1, -1)
                            else:
                                # Flatten the (H, C) dimensions to (H * C)
                                fp1_flat = _fp1.view(_fp1.size(0), -1)  # (B, H * C)
                                fp2_flat = _fp2.view(_fp2.size(0), -1)
                                fp2wr_flat = _fp2_wreuse.view(_fp2_wreuse.size(0), -1)
                            flatpoints += [fp1_flat, fp2_flat, fp2wr_flat]

                        # Concatenate the flattened tensors along the batch dimension
                        points = torch.cat(flatpoints, dim=0)

                        size1 = fp1_flat.size(0)
                        size2 = size1 + fp2_flat.size(0)
                        size3 = fp2wr_flat.size(0)
                        size1 *= samples_for_projection
                        size2 *= samples_for_projection
                        size3 *= samples_for_projection

                        # Convert to numpy for UMAP
                        data_np = points.detach().cpu().numpy()
                        print(f"Samples={data_np.shape[0]}, dim={data_np.shape[1]}")

                        # Apply UMAP for dimensionality reduction to 2D space
                        if batch_as_one and samples_for_projection == 1:
                            # RuntimeWarning: k >= N for N * N square matrix. Attempting to use scipy.linalg.eigh instead.
                            # Cannot use scipy.linalg.eigh for sparse A with k >= N. 
                            # Use scipy.linalg.eigh(A.toarray()) or reduce k
                            plot_dim = 1
                        else:
                            plot_dim = 2
                        reducer = umap.UMAP(n_components=plot_dim)
                        uemb = reducer.fit_transform(data_np)

                        reset_plot_styles()
                        set_seaborn_style()
                        fig, ax = plt.subplots()

                        colors = sns.color_palette(PALETTE).as_hex()
                        c0 = colors[0]
                        c1 = colors[1]
                        c2 = colors[2]
                        c0 = cdict["E4"]
                        c1 = cdict["E8"]
                        c2 = cdict["DEQ2"]

                        # plot the lines between the samples
                        # if samples_for_projection > 1:
                        #     # this draws lines between points of the same sample
                        #     for i in range(samples_for_projection):
                        #         plt.plot(
                        #             [uemb[i, 0], uemb[size1 + i, 0], uemb[size2 + i, 0]], 
                        #             [uemb[i, 1], uemb[size1 + i, 1], uemb[size2 + i, 1]],
                        #             marker="", 
                        #             # markersize=5, 
                        #             color="gray",
                        #             ls="-",
                        #             # very thin line
                        #             lw=0.5,
                        #         )

                        # Plot the results in 2D
                        plt.scatter(
                            uemb[:size1, 0], uemb[:size1, 1], label="FP 1", color=c0
                        )
                        plt.scatter(
                            uemb[size1:size2, 0], uemb[size1:size2, 1], label="FP 2", color=c1
                        )
                        plt.scatter(
                            uemb[size2:, 0], uemb[size2:, 1], label="FP 2 w reuse", color=c2
                        )

                        if not batch_as_one:
                            # get chemical symbols in the batch
                            symbols = [chemical_symbols[z] for z in data.z]
                            for i, txt in enumerate(symbols):
                                ax.annotate(txt, (uemb[i, 0], uemb[i, 1]))
                        

                        
                        # offset for the annotations
                        xoff = 0.03
                        yoff = 0.03
                        # annotate the samples
                        if samples_for_projection > 1 and samples_for_projection <= 20:
                            for i in range(samples_for_projection):
                                ax.annotate(
                                    f"{i}",
                                    (uemb[i, 0] + xoff, uemb[i, 1] + yoff),
                                    color=c0,
                                    fontsize=8,
                                    fontweight="bold",
                                )
                                ax.annotate(
                                    f"{i}",
                                    (uemb[size1 + i, 0] + xoff, uemb[size1 + i, 1] + yoff),
                                    color=c1,
                                    fontsize=8,
                                    fontweight="bold",
                                )
                                ax.annotate(
                                    f"{i}",
                                    (uemb[size2 + i, 0] + xoff, uemb[size2 + i, 1] + yoff),
                                    color=c2,
                                    fontsize=8,
                                    fontweight="bold",
                                )

                        plt.legend()
                        plt.title("UMAP projection of tensors")

                        set_style_after(ax=ax)

                        fname = "umap_fixedpoints"
                        if batch_as_one:
                            fname += "_batchasone"
                        fname += f"_{samples_for_projection}samples"
                        if consecutive:
                            fname += "_consecutive"
                        plt.savefig(f"{plotfolder}/{fname}.png")
                        print(f"Saved UMAP plot to\n {plotfolder}/{fname}.png")

                        plt.show()
                        
                        # break and finish
                        cnt = max_batches


                # reps finished

                # print statistics once we finished multiple repeats
                if max_repeats > 1 and rep == max_repeats - 1:
                    for reps, _name in zip(
                        [reps_fp1, reps_fp2, reps_fp2_wreuse],
                        ["fp1", "fp2", "fp2_wreuse"],
                    ):
                        if len(reps) > 1:
                            pass
                            # compute the pairwise distances between the fixed-points in the list
                            pairwise_distances = get_pairwise_distances(
                                reps
                            )  # TODO: replace this with a similarity matrix
                            # Idea: A UMAP embedding of the fixed-points?
                            pairwise_distances = torch.stack(pairwise_distances)

                            _mean = torch.mean(pairwise_distances).item()
                            _max = torch.max(pairwise_distances).item()
                            fp_repeat_avg_distances.append(_mean)
                            fp_repeat_max_distances.append(_max)

                            # only print if the distance is larger than the tolerance
                            if _max > repeat_tol:
                                # print the average distance
                                print(
                                    f"\n{cnt}: {max_repeats} repeats avg distance between {len(reps)} fixed-points ({_name}):",
                                    _mean,
                                )
                                # print the max distance
                                print(
                                    f"{cnt}: {max_repeats} repeats max distance between {len(reps)} fixed-points ({_name}):",
                                    _max,
                                )
                else:
                    pass
            
            if cnt >= max_batches - 1:
                break

            # test data points finished

        # no_grad ends

    # Summary statistics
    d_fp2_wo_reuse = torch.stack(d_fp2_wo_reuse)
    d_fp2_fp1 = torch.stack(d_fp2_fp1)
    d_fp2wreuse_fp1 = torch.stack(d_fp2wreuse_fp1)
    d_fp2_wo_reuse_rel = torch.stack(d_fp2_wo_reuse_rel)

    # Repeats
    if max_repeats > 1:
        fp_repeat_avg_distances = torch.stack(fp_repeat_avg_distances)
        fp_repeat_max_distances = torch.stack(fp_repeat_max_distances)
        print("\nRepeats: Do we always converge to the same fixed-points?")
        print(
            "Average distance between repeated fixed-points, averaged over data points:",
            torch.mean(fp_repeat_avg_distances).item(),
        )
        print(
            "Max     distance between repeated fixed-points, averaged over data points:",
            torch.mean(fp_repeat_max_distances).item(),
        )

    # Fixed-point reuse distance
    print(
        "\nAre fixed-points w/o reuse close? (i.e. does fixed-point reuse lead to different fixed-points?)"
    )
    print("Avg |fp2 - fp2_wreuse| =", torch.mean(d_fp2_wo_reuse).item())
    print(
        "Avg |fp2 - fp1|        =", torch.mean(d_fp2_fp1).item(), "<- should be larger"
    )
    print(
        "Avg |fp2_wreuse - fp1| =",
        torch.mean(d_fp2wreuse_fp1).item(),
        "<- should be larger",
    )
    _rel_err = torch.mean(d_fp2_wo_reuse / d_fp2_fp1).item()
    print("Avg |fp2 - fp2_wreuse| / |fp2 - fp1| =", _rel_err, "<- should be small")
    print(
        "Avg |fp2 - fp2_wreuse| / |fp2| =",
        torch.mean(d_fp2_wo_reuse_rel).item(),
        "<- should be small",
    )

    # plot the full distributions
    # df =

    # import seaborn as sns

    print("\nDone!")


if __name__ == "__main__":
    hydra_wrapper()
