import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from pathlib import Path
from typing import Iterable, Optional

import sys, os

# import equiformer.datasets.pyg.md17 as md17_dataset
import equiformer.datasets.pyg.md17revised as md17_dataset

from equiformer.logger import FileLogger

# import equiformer.nets
from equiformer.nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from equiformer.optim_factory import create_optimizer

from equiformer.engine import AverageMeter, compute_stats

from equiformer.nets.layer_norm import EquivariantLayerNormV2

import hydra
import wandb
import omegaconf
from omegaconf import DictConfig

from typing import (
    Union,
    Tuple,
    Any,
    Callable,
    Iterator,
    Set,
    Optional,
    overload,
    TypeVar,
    Mapping,
    Dict,
    List,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# nn.init.uniform_(self.source_embedding.weight.data, -0.001, 0.001)
# nn.init.uniform_(self.target_embedding.weight.data, -0.001, 0.001)
# nn.init.uniform_(self.weight, -np.sqrt(1 / d_hidden), np.sqrt(1 / d_hidden))


T = TypeVar("T", bound="Module")


class Module:
    # _modules: Dict[str, Optional['Module']]
    # named_children()

    def apply(self: T, fn: Callable[["Module"], None]) -> T:
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self."""
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self


def print_base_modules(module, print_without_params=False):
    children = [name for name, module in module.named_children()]
    if len(children) == 0:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if print_without_params or num_params > 0:
            print("module:", module)
            if isinstance(module, torch.nn.Linear):
                print("  -> linear")
            elif isinstance(module, torch.nn.LayerNorm):
                print("  -> LayerNorm")

            # some EquivariantLayerNormV2 are initialized to all 0s
            # some to all 1s
            # elif isinstance(module, EquivariantLayerNormV2):
            #     for p in module.parameters():
            #         print(p)

            # parameter lists are initialized to 0
            # elif isinstance(module, torch.nn.ParameterList):
            #     for p in module:
            #         print(p)
            else:
                # compute weight init
                try:
                    print(
                        "  weight:",
                        module.weight.data.mean().item(),
                        module.weight.data.std().item(),
                    )
                except:
                    print(f"  None ({num_params} params)")


def find_base_modules(model, args: DictConfig):

    model.apply(print_base_modules)


@hydra.main(
    config_name="md17",
    config_path="../equiformer/config/equiformer",
    version_base="1.3",
)
def hydra_wrapper(args: DictConfig) -> None:
    # print(train_idx)torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """ Dataset """
    train_dataset, val_dataset, test_dataset = md17_dataset.get_rmd17_datasets(
        root=os.path.join(args.data_path, args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        revised=args.md17revised,
        revised_old=args.md17revised_old,
        load_splits=args.use_revised_splits,
        order="consecutive_test" if args.fpreuse_test else None,
    )

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Network """
    create_model = model_entrypoint(args.model_name)
    model = create_model(task_mean=mean, task_std=std, **args.model_kwargs)

    find_base_modules(model, args)


if __name__ == "__main__":
    hydra_wrapper()
