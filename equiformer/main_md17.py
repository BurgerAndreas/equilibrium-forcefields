import argparse
import datetime
import itertools
import pickle
import subprocess
import time
import torch
import numpy as np
from torch_geometric.loader import DataLoader

import os
from pathlib import Path
from typing import Iterable, Optional

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import equiformer.datasets.pyg.md17 as md17_dataset

from equiformer.logger import FileLogger

# import equiformer.nets
from equiformer.nets import model_entrypoint

from timm.utils import ModelEmaV2, get_state_dict
from timm.scheduler import create_scheduler
from equiformer.optim_factory import create_optimizer

from equiformer.engine import AverageMeter, compute_stats

import hydra
import wandb
import omegaconf
from omegaconf import DictConfig

ModelEma = ModelEmaV2

"""
python main_md17.py \
    --output-dir 'models/md17/equiformer/se_l2/target@aspirin/lr@5e-4_wd@1e-6_epochs@1500_w-f2e@80_dropout@0.0_exp@32_l2mae-loss' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \ # None default
    --target 'aspirin' \
    --data-path 'datasets/md17' \
    --epochs 1500 \ # 1000 default
    --lr 5e-4 \
    --batch-size 8 \ 
    --weight-decay 1e-6 \ # 5e-3 default
    --num-basis 32 \ # 128 default
    --energy-weight 1 \ # 0.2 default
    --force-weight 80 # 0.8 default
"""


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser(
        "Training equivariant networks on MD17", add_help=False
    )
    parser.add_argument("--output-dir", type=str, default=None)
    # network architecture
    # graph_attention_transformer_nonlinear_exp_l2_md17
    # dot_product_attention_transformer_exp_l2_md17
    parser.add_argument(
        "--model-name",
        type=str,
        default="graph_attention_transformer_nonlinear_l2_md17",
    )
    parser.add_argument("--input-irreps", type=str, default=None)
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--num-basis", type=int, default=128)
    # training hyper-parameters
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)  # 8 -> 1
    parser.add_argument("--eval-batch-size", type=int, default=24)  # 24 -> 2
    parser.add_argument("--model-ema", action="store_true")
    parser.set_defaults(model_ema=False)
    parser.add_argument("--model-ema-decay", type=float, default=0.9999, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
    )
    # regularization
    parser.add_argument("--drop-path", type=float, default=0.0)
    # optimizer (timm)
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        # 1e-6
        "--weight-decay", type=float, default=5e-3, help="weight decay (default: 5e-3)"
    )
    # learning rate schedule parameters (timm)
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-6)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )
    # logging
    parser.add_argument("--print-freq", type=int, default=100)
    # task and dataset
    parser.add_argument("--target", type=str, default="aspirin")
    parser.add_argument("--data-path", type=str, default="datasets/md17")
    parser.add_argument("--train-size", type=int, default=950)
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--compute-stats", action="store_true", dest="compute_stats")
    parser.set_defaults(compute_stats=False)
    parser.add_argument(
        "--test-interval",
        type=int,
        default=10,
        help="epoch interval to evaluate on the testing set",
    )
    parser.add_argument(
        "--test-max-iter",
        type=int,
        default=1000,
        help="max iteration to evaluate on the testing set",
    )
    parser.add_argument("--energy-weight", type=float, default=0.2)
    parser.add_argument("--force-weight", type=float, default=0.8)
    # random
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=6)
    # data loader config
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)
    # evaluation
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--evaluate", action="store_true", dest="evaluate")
    # parser.add_argument("--meas_force", default=True, action="store_true")
    parser.add_argument("--meas_force", type=str2bool, nargs='?',
                        const=True, default=True,
                        help="Include force in loss calculation.")
    parser.set_defaults(evaluate=False)
    return parser


# from https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/modules/loss.py#L7
class L2MAELoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        assert reduction in ["mean", "sum"]

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        dists = torch.norm(input - target, p=2, dim=-1)
        if self.reduction == "mean":
            return torch.mean(dists)
        elif self.reduction == "sum":
            return torch.sum(dists)


def main(args, model=None):

    _log = FileLogger(is_master=True, is_rank0=True, output_dir=args.output_dir)
    _log.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    """ Dataset """
    train_dataset, val_dataset, test_dataset = md17_dataset.get_md17_datasets(
        root=os.path.join(args.data_path, args.target),
        dataset_arg=args.target,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=None,
        seed=args.seed,
    )

    _log.info("")
    _log.info("Training set size:   {}".format(len(train_dataset)))
    _log.info("Validation set size: {}".format(len(val_dataset)))
    _log.info("Testing set size:    {}\n".format(len(test_dataset)))

    # statistics
    y = torch.cat([batch.y for batch in train_dataset], dim=0)
    mean = float(y.mean())
    std = float(y.std())
    _log.info("Training set mean: {}, std: {}\n".format(mean, std))

    # since dataset needs random
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Network """
    create_model = model_entrypoint(args.model_name)
    model = create_model(
        irreps_in=args.input_irreps,
        radius=args.radius,
        num_basis=args.num_basis,
        task_mean=mean,
        task_std=std,
        atomref=None,
        drop_path=args.drop_path,
        num_layers=args.num_layers,
    )
    print(f"model {args.model_name} created")
    # _log.info(model)
    # else:
    #     model = model(
    #         irreps_in=args.input_irreps,
    #         max_radius=args.radius,
    #         number_of_basis=args.num_basis,
    #         mean=mean,
    #         std=std,
    #         atomref=None,
    #         drop_path=args.drop_path,
    #     )
    #     _log.info("Using the provided model")

    if args.checkpoint_path is not None:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict["state_dict"])

    model = model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else None,
        )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _log.info("Number of params: {}".format(n_parameters))

    """ Optimizer and LR Scheduler """
    optimizer = create_optimizer(args, model)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    criterion = (
        L2MAELoss()
    )  # torch.nn.L1Loss()  #torch.nn.MSELoss() # torch.nn.L1Loss()

    """ Data Loader """
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size)

    """ Compute stats """
    if args.compute_stats:
        compute_stats(
            train_loader,
            max_radius=args.radius,
            logger=_log,
            print_freq=args.print_freq,
        )
        return

    # record the best validation and testing errors and corresponding epochs
    best_metrics = {
        "val_epoch": 0,
        "test_epoch": 0,
        "val_force_err": float("inf"),
        "val_energy_err": float("inf"),
        "test_force_err": float("inf"),
        "test_energy_err": float("inf"),
    }
    best_ema_metrics = {
        "val_epoch": 0,
        "test_epoch": 0,
        "val_force_err": float("inf"),
        "val_energy_err": float("inf"),
        "test_force_err": float("inf"),
        "test_energy_err": float("inf"),
    }

    if args.evaluate:
        test_err, test_loss = evaluate(
            args=args,
            model=model,
            criterion=criterion,
            data_loader=test_loader,
            device=device,
            print_freq=args.print_freq,
            logger=_log,
            print_progress=True,
            max_iter=-1,
        )
        return

    global_step = 0
    for epoch in range(args.epochs):

        epoch_start_time = time.perf_counter()

        lr_scheduler.step(epoch)

        train_err, train_loss, global_step = train_one_epoch(
            args=args,
            model=model,
            criterion=criterion,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            model_ema=model_ema,
            print_freq=args.print_freq,
            logger=_log,
            meas_force=args.meas_force,
        )

        val_err, val_loss = evaluate(
            args=args,
            model=model,
            criterion=criterion,
            data_loader=val_loader,
            device=device,
            print_freq=args.print_freq,
            logger=_log,
            print_progress=False,
        )

        if (epoch + 1) % args.test_interval == 0:
            test_err, test_loss = evaluate(
                args=args,
                model=model,
                criterion=criterion,
                data_loader=test_loader,
                device=device,
                print_freq=args.print_freq,
                logger=_log,
                print_progress=True,
                max_iter=args.test_max_iter,
            )
        else:
            test_err, test_loss = None, None

        update_val_result, update_test_result = update_best_results(
            args, best_metrics, val_err, test_err, epoch
        )
        if update_val_result:
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(
                    args.output_dir,
                    "best_val_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, val_err["energy"].avg, val_err["force"].avg
                    ),
                ),
            )
        if update_test_result:
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(
                    args.output_dir,
                    "best_test_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, test_err["energy"].avg, test_err["force"].avg
                    ),
                ),
            )
        if (
            (epoch + 1) % args.test_interval == 0
            and (not update_val_result)
            and (not update_test_result)
        ):
            torch.save(
                {"state_dict": model.state_dict()},
                os.path.join(
                    args.output_dir,
                    "epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                        epoch, test_err["energy"].avg, test_err["force"].avg
                    ),
                ),
            )

        # log once per epoch
        info_str = "Epoch: [{epoch}] Target: [{target}] train_e_MAE: {train_e_mae:.5f}, train_f_MAE: {train_f_mae:.5f}, ".format(
            epoch=epoch,
            target=args.target,
            train_e_mae=train_err["energy"].avg,
            train_f_mae=train_err["force"].avg,
        )
        info_str += "val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, ".format(
            val_err["energy"].avg, val_err["force"].avg
        )
        if (epoch + 1) % args.test_interval == 0:
            info_str += "test_e_MAE: {:.5f}, test_f_MAE: {:.5f}, ".format(
                test_err["energy"].avg, test_err["force"].avg
            )
        info_str += "Time: {:.2f}s".format(time.perf_counter() - epoch_start_time)
        _log.info(info_str)

        # log to wandb
        logs = {
                "train_e_mae": train_err["energy"].avg,
                "train_f_mae": train_err["force"].avg,
                "val_e_mae": val_err["energy"].avg,
                "val_f_mae": val_err["force"].avg,
                "lr": optimizer.param_groups[0]["lr"],
                # allows us to plot against epoch
                # in the custom plots, click edit and select a custom x-axis
                "epoch": epoch, 
            }
        if test_err is not None:
            logs["test_e_mae"] = test_err["energy"].avg
            logs["test_f_mae"] = test_err["force"].avg
        wandb.log(
            logs,
            # step=epoch,
            step=global_step,
        )

        info_str = "Best -- val_epoch={}, test_epoch={}, ".format(
            best_metrics["val_epoch"], best_metrics["test_epoch"]
        )
        info_str += "val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, ".format(
            best_metrics["val_energy_err"], best_metrics["val_force_err"]
        )
        info_str += "test_e_MAE: {:.5f}, test_f_MAE: {:.5f}\n".format(
            best_metrics["test_energy_err"], best_metrics["test_force_err"]
        )
        _log.info(info_str)

        # log to wandb
        wandb.log(
            {
                "best_val_e_mae": best_metrics["val_energy_err"],
                "best_val_f_mae": best_metrics["val_force_err"],
                "best_test_e_mae": best_metrics["test_energy_err"],
                "best_test_f_mae": best_metrics["test_force_err"],
            },
            # step=epoch,
            step=global_step,
        )

        # evaluation with EMA
        if model_ema is not None:
            ema_val_err, _ = evaluate(
                args=args,
                model=model_ema.module,
                criterion=criterion,
                data_loader=val_loader,
                device=device,
                print_freq=args.print_freq,
                logger=_log,
                print_progress=False,
            )

            if (epoch + 1) % args.test_interval == 0:
                ema_test_err, _ = evaluate(
                    args=args,
                    model=model_ema.module,
                    criterion=criterion,
                    data_loader=test_loader,
                    device=device,
                    print_freq=args.print_freq,
                    logger=_log,
                    print_progress=True,
                    max_iter=args.test_max_iter,
                )
            else:
                ema_test_err, ema_test_loss = None, None

            update_val_result, update_test_result = update_best_results(
                args, best_ema_metrics, ema_val_err, ema_test_err, epoch
            )

            if update_val_result:
                torch.save(
                    {"state_dict": get_state_dict(model_ema)},
                    os.path.join(
                        args.output_dir,
                        "best_ema_val_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                            epoch, ema_val_err["energy"].avg, ema_val_err["force"].avg
                        ),
                    ),
                )
            if update_test_result:
                torch.save(
                    {"state_dict": get_state_dict(model_ema)},
                    os.path.join(
                        args.output_dir,
                        "best_ema_test_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                            epoch, ema_test_err["energy"].avg, ema_test_err["force"].avg
                        ),
                    ),
                )
            if (
                (epoch + 1) % args.test_interval == 0
                and (not update_val_result)
                and (not update_test_result)
            ):
                torch.save(
                    {"state_dict": get_state_dict(model_ema)},
                    os.path.join(
                        args.output_dir,
                        "ema_epochs@{}_e@{:.4f}_f@{:.4f}.pth.tar".format(
                            epoch, test_err["energy"].avg, test_err["force"].avg
                        ),
                    ),
                )

            info_str = "EMA "
            info_str += "val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, ".format(
                ema_val_err["energy"].avg, ema_val_err["force"].avg
            )
            wandb.log(
                {
                    "EMA_val_e_mae": ema_val_err["energy"].avg,
                    "EMA_val_f_mae": ema_val_err["force"].avg,
                },
                # step=epoch,
                step=global_step,
            )

            if (epoch + 1) % args.test_interval == 0:
                info_str += "test_e_MAE: {:.5f}, test_f_MAE: {:.5f}, ".format(
                    ema_test_err["energy"].avg, ema_test_err["force"].avg
                )
                wandb.log(
                    {
                        "EMA_test_e_mae": ema_test_err["energy"].avg,
                        "EMA_test_f_mae": ema_test_err["force"].avg,
                    },
                    # step=epoch,
                    step=global_step,
                )

            info_str += "Time: {:.2f}s".format(time.perf_counter() - epoch_start_time)
            _log.info(info_str)

            info_str = "Best EMA -- val_epoch={}, test_epoch={}, ".format(
                best_ema_metrics["val_epoch"], best_ema_metrics["test_epoch"]
            )
            info_str += "val_e_MAE: {:.5f}, val_f_MAE: {:.5f}, ".format(
                best_ema_metrics["val_energy_err"], best_ema_metrics["val_force_err"]
            )
            info_str += "test_e_MAE: {:.5f}, test_f_MAE: {:.5f}\n".format(
                best_ema_metrics["test_energy_err"], best_ema_metrics["test_force_err"]
            )
            _log.info(info_str)

            # log to wandb
            wandb.log(
                {
                    "EMA_best_val_e_mae": best_ema_metrics["val_energy_err"],
                    "EMA_best_val_f_mae": best_ema_metrics["val_force_err"],
                    "EMA_best_test_e_mae": best_ema_metrics["test_energy_err"],
                    "EMA_best_test_f_mae": best_ema_metrics["test_force_err"],
                },
                # step=epoch,
                step=global_step,
            )


    # evaluate on the whole testing set
    test_err, test_loss = evaluate(
        args=args,
        model=model,
        criterion=criterion,
        data_loader=test_loader,
        device=device,
        print_freq=args.print_freq,
        logger=_log,
        print_progress=True,
        max_iter=-1,
    )


def update_best_results(args, best_metrics, val_err, test_err, epoch):
    def _compute_weighted_error(args, energy_err, force_err):
        return args.energy_weight * energy_err + args.force_weight * force_err

    update_val_result, update_test_result = False, False

    print(f'Trying to update best results for epoch {epoch}')
    new_loss = _compute_weighted_error(
        args, val_err["energy"].avg, val_err["force"].avg
    )
    prev_loss = _compute_weighted_error(
        args, best_metrics["val_energy_err"], best_metrics["val_force_err"]
    )
    print(f' New loss val: {new_loss}, prev loss: {prev_loss}')
    if new_loss < prev_loss:
        best_metrics["val_energy_err"] = val_err["energy"].avg
        best_metrics["val_force_err"] = val_err["force"].avg
        best_metrics["val_epoch"] = epoch
        update_val_result = True

    if test_err is None:
        print(f' Test error is None, skipping updating best val for epoch {epoch}')
        return update_val_result, update_test_result

    new_loss = _compute_weighted_error(
        args, test_err["energy"].avg, test_err["force"].avg
    )
    prev_loss = _compute_weighted_error(
        args, best_metrics["test_energy_err"], best_metrics["test_force_err"]
    )
    print(f' New loss test: {new_loss}, prev loss: {prev_loss}')
    if new_loss < prev_loss:
        best_metrics["test_energy_err"] = test_err["energy"].avg
        best_metrics["test_force_err"] = test_err["force"].avg
        best_metrics["test_epoch"] = epoch
        update_test_result = True

    return update_val_result, update_test_result


def train_one_epoch(
    args,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    model_ema: Optional[ModelEma] = None,
    print_freq: int = 100,
    logger=None,
    meas_force=True,
):

    model.train()
    criterion.train()

    loss_metrics = {"energy": AverageMeter(), "force": AverageMeter()}
    mae_metrics = {"energy": AverageMeter(), "force": AverageMeter()}

    start_time = time.perf_counter()

    task_mean = model.task_mean
    task_std = model.task_std

    # z_star = None
    for step, data in enumerate(data_loader):
        data = data.to(device)

        # energy, force
        pred_y, pred_dy = model(node_atom=data.z, pos=data.pos, batch=data.batch)
        # if deq_mode and reuse:
        #     z_star = z_pred.detach()

        loss_e = criterion(pred_y, ((data.y - task_mean) / task_std))
        loss = args.energy_weight * loss_e
        if meas_force == True:
            loss_f = criterion(pred_dy, (data.dy / task_std))
            loss += args.force_weight * loss_f

        # If you use trajectory sampling, fp_correction automatically
        # aligns the tensors and applies your loss function.
        # loss_fn = lambda y_gt, y: ((y_gt - y) ** 2).mean()
        # train_loss = fp_correction(loss_fn, (y_train, y_pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_metrics["energy"].update(loss_e.item(), n=pred_y.shape[0])
        loss_metrics["force"].update(loss_f.item(), n=pred_dy.shape[0])

        energy_err = pred_y.detach() * task_std + task_mean - data.y
        energy_err = torch.mean(torch.abs(energy_err)).item()
        mae_metrics["energy"].update(energy_err, n=pred_y.shape[0])
        force_err = pred_dy.detach() * task_std - data.dy
        force_err = torch.mean(
            torch.abs(force_err)
        ).item()  # based on OC20 and TorchMD-Net, they average over x, y, z
        mae_metrics["force"].update(force_err, n=pred_dy.shape[0])

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        # logging
        if step % print_freq == 0 or step == len(data_loader) - 1:
            w = time.perf_counter() - start_time
            e = (step + 1) / len(data_loader)
            info_str = "Epoch: [{epoch}][{step}/{length}] \t".format(
                epoch=epoch, step=step, length=len(data_loader)
            )
            info_str += "loss_e: {loss_e:.5f}, loss_f: {loss_f:.5f}, e_MAE: {e_mae:.5f}, f_MAE: {f_mae:.5f}, ".format(
                loss_e=loss_metrics["energy"].avg,
                loss_f=loss_metrics["force"].avg,
                e_mae=mae_metrics["energy"].avg,
                f_mae=mae_metrics["force"].avg,
            )
            info_str += "time/step={time_per_step:.0f}ms, ".format(
                time_per_step=(1e3 * w / e / len(data_loader))
            )
            info_str += "lr={:.2e}".format(optimizer.param_groups[0]["lr"])
            logger.info(info_str)

            # log to wandb
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_e_mae": mae_metrics["energy"].avg,
                    "train_f_mae": mae_metrics["force"].avg,
                    "lr": optimizer.param_groups[0]["lr"],
                },
                step=global_step,
            )

        global_step += 1

    return mae_metrics, loss_metrics, global_step


def evaluate(
    args,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    print_freq: int = 100,
    logger=None,
    print_progress=False,
    max_iter=-1,
):

    model.eval()
    criterion.eval()
    loss_metrics = {"energy": AverageMeter(), "force": AverageMeter()}
    mae_metrics = {"energy": AverageMeter(), "force": AverageMeter()}

    start_time = time.perf_counter()

    task_mean = model.task_mean
    task_std = model.task_std

    with torch.no_grad():

        for step, data in enumerate(data_loader):

            data = data.to(device)
            pred_y, pred_dy = model(node_atom=data.z, pos=data.pos, batch=data.batch)

            loss_e = criterion(pred_y, ((data.y - task_mean) / task_std))
            loss_f = criterion(pred_dy, (data.dy / task_std))

            loss_metrics["energy"].update(loss_e.item(), n=pred_y.shape[0])
            loss_metrics["force"].update(loss_f.item(), n=pred_dy.shape[0])

            energy_err = pred_y.detach() * task_std + task_mean - data.y
            energy_err = torch.mean(torch.abs(energy_err)).item()
            mae_metrics["energy"].update(energy_err, n=pred_y.shape[0])
            force_err = pred_dy.detach() * task_std - data.dy
            force_err = torch.mean(
                torch.abs(force_err)
            ).item()  # based on OC20 and TorchMD-Net, they average over x, y, z
            mae_metrics["force"].update(force_err, n=pred_dy.shape[0])

            # logging
            if (
                step % print_freq == 0 or step == len(data_loader) - 1
            ) and print_progress:
                w = time.perf_counter() - start_time
                e = (step + 1) / len(data_loader)
                info_str = "[{step}/{length}] \t".format(
                    step=step, length=len(data_loader)
                )
                info_str += "e_MAE: {e_mae:.5f}, f_MAE: {f_mae:.5f}, ".format(
                    e_mae=mae_metrics["energy"].avg,
                    f_mae=mae_metrics["force"].avg,
                )
                info_str += "time/step={time_per_step:.0f}ms".format(
                    time_per_step=(1e3 * w / e / len(data_loader))
                )
                logger.info(info_str)

            if ((step + 1) >= max_iter) and (max_iter != -1):
                break

    return mae_metrics, loss_metrics



@hydra.main(config_name="md17", config_path="config/equiformer", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop.
    
    Usage:
        python deq_equiformer.py
        python deq_equiformer.py batch_size=8
        python deq_equiformer.py +machine=vector

    Usage with slurm:
        sbatch scripts/slurm_launcher.slrm deq_equiformer.py +machine=vector
    """

    # graph_attention_transformer_nonlinear_exp_l2_md17
    # dot_product_attention_transformer_exp_l2_md17
    # args.model_name = "graph_attention_transformer_nonlinear_exp_l2_md17"

    args.output_dir = "models/md17/equiformer/se_l2/target@aspirin/lr@5e-4_wd@1e-6_epochs@1500_w-f2e@80_dropout@0.0_exp@32_l2mae-loss"
    args.input_irreps = "64x0e"
    args.target = "aspirin"
    args.data_path = "datasets/md17"
    args.epochs = 1500
    args.lr = 5e-4
    args.batch_size = 1
    args.eval_batch_size = 2
    args.weight_decay = 1e-6
    args.num_basis = 32
    args.energy_weight = 1
    args.force_weight = 80

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    from deq_equiformer.logging_utils import init_wandb
    init_wandb(args)

    main(args)

if __name__ == "__main__":

    # parser = argparse.ArgumentParser(
    #     "Training equivariant networks on MD17", parents=[get_args_parser()]
    # )
    # args = parser.parse_args()

    # graph_attention_transformer_nonlinear_exp_l2_md17
    # dot_product_attention_transformer_exp_l2_md17
    
    hydra_wrapper()