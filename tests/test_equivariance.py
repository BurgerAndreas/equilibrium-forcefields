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

@hydra.main(
    config_name="md17", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    # also load the args from equiformer/config/md17.yaml
    # and update the args with the new args (if not already present)
    # args = OmegaConf.merge(args, OmegaConf.load("equiformer/config/md17.yaml"))
    # argsmd17 = OmegaConf.load("equiformer/config/md17.yaml")
    # argsmd17.update(args)
    # args = argsmd17

    
    args.return_model_and_data = True
    # args.model.max_num_elements = 10

    # init_wandb(args, project="equilibrium-forcefields-equiformer_v2")
    run_id = init_wandb(args)

    datas = train(args)
    model = datas["model"]
    train_dataset = datas["train_dataset"]
    test_dataset_full = datas["test_dataset_full"]
    # device = model.device
    device = list(model.parameters())[0].device

    collate = Collater(follow_batch=None, exclude_keys=None)
    equivariance_test(args, model, train_dataset, test_dataset_full, device, collate)
    

    print('\nDone!')


if __name__ == "__main__":
    hydra_wrapper()