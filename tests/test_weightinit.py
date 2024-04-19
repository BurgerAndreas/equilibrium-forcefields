import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from pathlib import Path
from typing import Iterable, Optional

import sys, os

# import equiformer.datasets.pyg.md17 as md17_dataset
import equiformer.datasets.pyg.md_all as md17_dataset

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

import deq2ff
from deq2ff.deq_equiformer.deq_dp_md17 import (
    deq_dot_product_attention_transformer_exp_l2_md17,
)
from deq2ff.deq_equiformer.deq_graph_md17 import (
    deq_graph_attention_transformer_nonlinear_l2_md17,
)
from deq2ff.deq_equiformer.deq_dp_md17_noforce import (
    deq_dot_product_attention_transformer_exp_l2_md17_noforce,
)
from deq2ff.deq_equiformer.deq_dp_minimal import deq_minimal_dpa

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


# coloring

import colorama
from termcolor import colored

colorama.init()


def print_base_modules(module, print_without_params=False, bias=True):
    children = [name for name, module in module.named_children()]

    if len(children) == 0:
        num_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        if print_without_params or num_params > 0:
            print(module)
            if isinstance(module, torch.nn.Linear):
                # print("  -> linear")
                print(
                    f"  weight: mean={module.weight.data.mean().item()}, std={module.weight.data.std().item()}, nonzeros={module.weight.data.nonzero().size(0)}"
                )
                if module.bias is None:
                    print("  bias: None")
                else:
                    text = f"  bias: mean={module.bias.data.mean().item()}, std={module.bias.data.std().item()}"
                    if bias:
                        print(text)
                    else:
                        print(colored(text, "black", "on_green"))

            # LayerNorm is initialized to weight=1, bias=0
            elif isinstance(module, torch.nn.LayerNorm):
                # print("  -> LayerNorm")
                print(
                    f"  weight: mean={module.weight.data.mean().item()}, std={module.weight.data.std().item()}, nonzeros={module.weight.data.nonzero().size(0)}"
                )
                if module.bias is None:
                    print("  bias: None")
                else:
                    text = f"  bias: mean={module.bias.data.mean().item()}, std={module.bias.data.std().item()}"
                    if bias:
                        print(text)
                    else:
                        print(
                            colored(text, "black", "on_green")
                        )  # bias shouldn't appear

            # EquivariantLayerNormV2 are initialized to weight=0, bias=1
            elif isinstance(module, EquivariantLayerNormV2):
                # print("  -> EquivariantLayerNormV2")
                num_zeros = (module.affine_weight.data == 0).sum().item()
                num_ones = (module.affine_weight.data == 1).sum().item()
                print(f"  weight: zeros={num_zeros}, ones={num_ones}")
                if module.affine is False:
                    print("  bias: None")
                else:
                    num_zeros = (module.affine_bias.data == 0).sum().item()
                    num_ones = (module.affine_bias.data == 1).sum().item()
                    text = f"  bias: zeros={num_zeros}, ones={num_ones}"
                    if bias:
                        print(text)
                    else:
                        print(colored(text, "black", "on_red"))  # bias shouldn't appear
                # for p in module.parameters():
                #     print(p)

            # ParameterList are initialized to 0
            elif isinstance(module, torch.nn.ParameterList):
                mean = torch.stack([p.data.mean() for p in module]).mean().item()
                std = torch.stack([p.data.std() for p in module]).mean().item()
                nonzeros = [p.data.nonzero().size(0) for p in module][0]
                # print("  -> ParameterList")
                print(f"  weight: mean={mean}, std={std}, nonzeros={nonzeros}")
                # for p in module:
                #     print(p)

            else:
                # compute weight init
                try:
                    print(
                        f"  weight: mean={module.weight.data.mean().item()}, std={module.weight.data.std().item()}, nonzeros={module.weight.data.nonzero().size(0)}"
                    )
                except:
                    print(f"  None ({num_params} params)")


def find_base_modules(model, args: DictConfig):

    # model.apply(print_base_modules)
    model.blocks.apply(print_base_modules)  # deq implicit layer


"""
Things to check:
- bias=False
- 
"""


@hydra.main(
    config_name="deq",  # md17
    config_path="../equiformer/config",
    version_base="1.3",
)
def hydra_wrapper(args: DictConfig) -> None:
    # print(train_idx)torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    """ Dataset """
    train_dataset, val_dataset, test_dataset = md17_dataset.get_md_datasets(
        root=os.path.join(args.data_path, "md17", args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
        dname=args.dname,
        load_splits=args.use_revised_splits,
        order="consecutive_test" if args.fpreuse_test else None,
    )

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Change kwargs """
    # args.model.bias = False
    args.model.deq_block = "FFNormResidual"
    # args.model.weight_init_blocks={'EquivariantLayerNormV2_b':0}
    # args.model.weight_init_blocks={'EquivariantLayerNormV2_w':1,'EquivariantLayerNormV2_b':0}

    """ Network """
    # create_model = model_entrypoint(args.model.name)
    model_name = "deq_" + args.model.name
    model_name = "deq_minimal_dpa"

    create_model = model_entrypoint(model_name)
    model = create_model(task_mean=mean, task_std=std, **args.model)

    """ Print base modules """
    find_base_modules(model, args)

    # print("\nmodel:\n", model)
    print("\nImplicit layer:\n", model.blocks)


if __name__ == "__main__":
    hydra_wrapper()
