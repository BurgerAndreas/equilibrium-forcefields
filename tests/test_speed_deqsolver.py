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
from scripts.train_deq_md import train_md, get_normalizers, load_loss

# register all models
import deq2ff.register_all_models

# profiling
from torch.profiler import profile, record_function, ProfilerActivity

file_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(file_dir)

"""
use with:
+use=deq +cfg=ap
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

    # args.test_patch_size = 2 # needs to be an even number, s.t. we can use fpreuse every second datadpoint
    # args.test_patches = 100 # the more the longer. 10 to 10000. Default: 1000

    #############################

    # get data
    args.return_model_and_data = True
    # ensure we load a checkpoint of a trained model
    # args.assert_checkpoint = True


    # init_wandb(args, project="equilibrium-forcefields-equiformer_v2")
    args.wandb = False # TODO: we are not logging anything
    run_id = init_wandb(args)

    datas = train_md(args)
    model = datas["model"]
    train_dataset = datas["train_dataset"]
    train_loader = datas["train_loader"]
    # test_dataset_full = datas["test_dataset_full"]
    # test_dataset = datas["test_dataset"]
    test_loader = datas["test_loader"]
    optimizer = datas["optimizer"]
    
    device = list(model.parameters())[0].device
    dtype = model.parameters().__next__().dtype

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    task_mean = float(y.mean())
    task_std = float(y.std())    
    normalizers = get_normalizers(args, train_dataset, device, task_mean, task_std)

    # criterion = L2MAELoss()
    loss_fn = load_loss({"energy": args.loss_energy, "force": args.loss_force})
    criterion_energy = loss_fn["energy"]
    criterion_force = loss_fn["force"]

    # eval mode
    model.train()
    
    ######################################
    if args.torch_profile:
        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step
        prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
        )
        prof.start()

    ######################################
    # loop forward
    for batchstep, data in enumerate(train_loader):
        wandb.log({"memalloc-pre_batch": torch.cuda.memory_allocated()}, step=batchstep)
        data = data.to(device)
        data = data.to(device, dtype)

        pred_y, pred_dy, info = model(
            data=data,  # for EquiformerV2
            # for EquiformerV1:
            # node_atom=data.z,
            # pos=data.pos,
            # batch=data.batch,
            # step=global_step,
            datasplit="train",
            fpr_loss=args.fpr_loss,
        )

        if args.torch_profile:
            prof.step()
        
    # end of epoch
    if args.torch_profile:
        prof.stop()
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        # print location of the trace file
        mname = args.checkpoint_wandb_name  # wandb.run.name
        # remove special characters
        mname = "".join(e for e in mname if e.isalnum())
        prof.export_chrome_trace(f"{parent_dir}/traces/{mname}.json")
        print("Saved trace to:", f"{parent_dir}/traces/{mname}.json")
        exit()

    print('\nDone!')


if __name__ == "__main__":
    hydra_wrapper()