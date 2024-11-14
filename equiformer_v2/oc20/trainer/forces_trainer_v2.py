f"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.


Copy force trainer from https://github.com/Open-Catalyst-Project/ocp/tree/6cd108e95f006a268f19459ca1b5ec011749da37

"""


import os
import pathlib
from collections import defaultdict
from pathlib import Path
import yaml
import time
import pprint

import numpy as np
import torch
import torch_geometric
from tqdm import tqdm
import math as math

from ocpmodels.common import distutils
from ocpmodels.common.registry import registry
from ocpmodels.common.relaxation.ml_relaxation import ml_relax
from ocpmodels.common.utils import check_traj_files
from ocpmodels.modules.evaluator import Evaluator
from ocpmodels.modules.normalizer import Normalizer

# from ocpmodels.trainers.base_trainer import BaseTrainer
from .base_trainer_v2 import BaseTrainerV2
from .engine import AverageMeter

import deq2ff.logging_utils_deq as logging_utils_deq


@registry.register_trainer("forces_v2")
class ForcesTrainerV2(BaseTrainerV2):
    """
    Trainer class for the Structure to Energy & Force (S2EF) and Initial State to
    Relaxed State (IS2RS) tasks.

    .. note::

        Examples of configurations for task, model, dataset and optimizer
        can be found in `configs/ocp_s2ef <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2re/>`_
        and `configs/ocp_is2rs <https://github.com/Open-Catalyst-Project/baselines/tree/master/configs/ocp_is2rs/>`_.

    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        is_hpo (bool, optional): Run hyperparameter optimization with Ray Tune.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`tensorboard`)
        local_rank (int, optional): Local rank of the process, only applicable for distributed training.
            (default: :obj:`0`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        checkpoint_name=None,
        checkpoint_path=None,
        load_checkpoint=True,
        checkpoint_wandb_name=None,
        assert_checkpoint=False,
        is_debug=False,
        is_hpo=False,
        print_every=100,
        seed=0,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
        # added
        val_max_iter=-1,
        test_w_eval_mode=True,
        skip_dataset=False,
        **kwargs,
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            checkpoint_name=checkpoint_name,
            checkpoint_wandb_name=checkpoint_wandb_name,
            checkpoint_path=checkpoint_path,
            load_checkpoint=load_checkpoint,
            assert_checkpoint=assert_checkpoint,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="s2ef",
            slurm=slurm,
            noddp=noddp,
            # added
            val_max_iter=val_max_iter,
            test_w_eval_mode=test_w_eval_mode,
            skip_dataset=skip_dataset,
            **kwargs,
        )
        self.fixedpoint = None
        self.fpreuse_test = False
        
    def load_task(self):
        self.file_logger.info(f"Loading dataset: {self.config['task']['dataset']}")

        if "relax_dataset" in self.config["task"]:

            self.relax_dataset = registry.get_dataset_class(
                self.config["task"]["dataset"]
            )(self.config["task"]["relax_dataset"])
            self.relax_sampler = self.get_sampler(
                self.relax_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.relax_loader = self.get_dataloader(
                self.relax_dataset,
                self.relax_sampler,
            )

        self.num_targets = 1

        # If we're computing gradients wrt input, set mean of normalizer to 0 --
        # since it is lost when compute dy / dx -- and std to forward target std
        if self.config["model_attributes"].get("regress_forces", True) or self.config[
            "model_attributes"
        ].get("use_auxiliary_task", False):
            if self.normalizer.get("normalize_labels", False):
                if "grad_target_mean" in self.normalizer:
                    self.normalizers["grad_target"] = Normalizer(
                        mean=self.normalizer["grad_target_mean"],
                        std=self.normalizer["grad_target_std"],
                        device=self.device,
                    )
                else:
                    self.normalizers["grad_target"] = Normalizer(
                        tensor=self.train_loader.dataset.data.y[
                            self.train_loader.dataset.__indices__
                        ],
                        device=self.device,
                    )
                    self.normalizers["grad_target"].mean.fill_(0)

    def init_prediction_metrics(self):
        self.prediction_metrics = {
            "loss": [],
            "nstep": [],
        }
    
    # Takes in a new data source and generates predictions on it.
    @torch.no_grad()
    def predict(
        self,
        data_loader,
        per_image=True,
        results_file=None,
        disable_tqdm=False,
    ):
        """
        Predicts on a given dataset.
        Not run during training, only for evaluation.

        Args:
        per image (bool): ?
        """
        if per_image:
            self.file_logger.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            data_loader = [[data_loader]]

        self.model.eval()
        # Copy EMA parameters to model.
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        if self.normalizers is not None and "target" in self.normalizers:
            self.normalizers["target"].to(self.device)
            self.normalizers["grad_target"].to(self.device)

        predictions = {"id": [], "energy": [], "forces": [], "chunk_idx": []}

        for i, batch_list in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc="device {}".format(rank),
            disable=disable_tqdm,
        ):
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch_list, fpreuse=self.fpreuse_test)

            if self.normalizers is not None and "target" in self.normalizers:
                out["energy"] = self.normalizers["target"].denorm(out["energy"])
                out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])
                
            if per_image: # False by default
                systemids = [
                    str(i) + "_" + str(j)
                    for i, j in zip(
                        batch_list[0].sid.tolist(), batch_list[0].fid.tolist()
                    )
                ]
                predictions["id"].extend(systemids)
                predictions["energy"].extend(out["energy"].to(torch.float16).tolist())
                batch_natoms = torch.cat([batch.natoms for batch in batch_list])
                batch_fixed = torch.cat([batch.fixed for batch in batch_list])
                forces = out["forces"].cpu().detach().to(torch.float16)
                per_image_forces = torch.split(forces, batch_natoms.tolist())
                per_image_forces = [force.numpy() for force in per_image_forces]
                # evalAI only requires forces on free atoms
                if results_file is not None:
                    _per_image_fixed = torch.split(batch_fixed, batch_natoms.tolist())
                    _per_image_free_forces = [
                        force[(fixed == 0).tolist()]
                        for force, fixed in zip(per_image_forces, _per_image_fixed)
                    ]
                    _chunk_idx = np.array(
                        [free_force.shape[0] for free_force in _per_image_free_forces]
                    )
                    per_image_forces = _per_image_free_forces
                    predictions["chunk_idx"].extend(_chunk_idx)
                predictions["forces"].extend(per_image_forces)
            
            else:
                predictions["energy"] = out["energy"].detach()
                predictions["forces"] = out["forces"].detach()
                
                self.log_nstep(out)

                if self.ema:
                    self.ema.restore()
                    
                return predictions

        predictions["forces"] = np.array(predictions["forces"])
        predictions["chunk_idx"] = np.array(predictions["chunk_idx"])
        predictions["energy"] = np.array(predictions["energy"])
        predictions["id"] = np.array(predictions["id"])
        self.save_results(
            predictions, results_file, keys=["energy", "forces", "chunk_idx"]
        )
        
        self.log_nstep(out)

        if self.ema:
            self.ema.restore()

        return predictions
    
    def log_nstep(self, out):
        # logging
        info = out["info"]
        if "nstep" in info:
            nstep = info["nstep"].mean().item()
            self.prediction_metrics["nstep"].append(nstep)
            self.logger.log({"nstep": nstep}, step=self.step)

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm=True,
    ):
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )

    def train(self, disable_eval_tqdm=False):
        # pretty print the config.

        pp = pprint.PrettyPrinter(depth=4)
        self.file_logger.info(f"ForcesTrainerV2: self.config:")
        pp.pprint(self.config)
        self.file_logger.info(f"------------------------------")
        self.logger.config.update(self.config)
        self.file_logger.info(f"Size training set: {len(self.train_loader)}")
        self.logger.log({"num_training_steps": len(self.train_loader)}, step=self.step)

        # eval at least once per epoch
        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.config["task"].get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if not hasattr(self, "primary_metric") or self.primary_metric != primary_metric:
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)

        # tqdm might not work well with distributed training.
        num_steps = len(self.train_loader) * self.config["optim"]["max_epochs"]

        isnan_cnt = 0

        def get_pbar_desc(epoch_int):
            return f"Train (device {distutils.get_rank()}) Epoch={epoch_int}"

        pbar = tqdm(
            total=num_steps,
            position=distutils.get_rank(),
            # desc="Train (device {})".format(distutils.get_rank()),
            desc=get_pbar_desc(start_epoch),
            # desc="Training",
            unit="steps",
        )

        # Only take a subset of the training data without changing the dataloader
        if self.maxdata > 0:
            max_steps = min(
                len(self.train_loader),
                self.maxdata // self.config["optim"]["batch_size"],
            )
        else:
            max_steps = len(self.train_loader)

        self.file_logger.info(f"Starting training at step {self.step}.")
        self.file_logger.info(f"Steps per epoch: {max_steps}")
        self.logger.log(
            {
                "max_steps": int(max_steps * self.config["optim"]["max_epochs"]),
                "steps_per_epoch": max_steps,
            },
            split="train",
            step=self.step,
        )
        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            self.train_sampler.set_epoch(epoch_int)
            self.logger.log({"epoch": self.epoch}, step=self.step)
            skip_steps = self.step % len(self.train_loader)
            train_loader_iter = iter(self.train_loader)

            # reinitialize metrics
            self.metrics = {}

            self.model.train()
            # loop over batches
            for i in range(skip_steps, max_steps):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1

                # Get a batch.
                batch = next(train_loader_iter)

                # Forward, loss, backward.
                with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)
                loss = self.scaler.scale(loss) if self.scaler else loss
                if self.grad_accumulation_steps != 1:
                    loss = loss / self.grad_accumulation_steps
                # calls optimizer.step and ema.update
                self._backward(loss)
                scale = self.scaler.get_scale() if self.scaler else 1.0

                # Compute metrics.
                self.metrics, results = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                    return_results=True,
                )
                self.metrics = self.evaluator.update(
                    "loss",
                    loss.item() / scale * self.grad_accumulation_steps,
                    self.metrics,
                )

                # Log the AVERAGED metrics, not the per-batch metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update({f"{k}_batch": results[k]["metric"] for k in results})
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                        "loss_batch": loss.item()
                        / scale
                        * self.grad_accumulation_steps,
                    }
                )

                # DEQ logging
                info = out["info"]
                if "nstep" in info:
                    log_dict["nstep"] = info["nstep"].mean().item()
                    
                if "abs_trace" in info.keys():
                    # log fixed-point trajectory once per evaluation
                    logging_utils_deq.log_fixed_point_error(
                        info,
                        step=self.step,
                        datasplit="train",
                    )

                # logging of losses
                if (
                    (
                        self.step % self.config["cmd"]["print_every"] == 0
                        or i == 0
                        or i == (len(self.train_loader) - 1)
                    )
                    and distutils.is_master()
                    and not self.is_hpo
                ):
                    log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                    # self.file_logger.info(", ".join(log_str))
                    # self.metrics = {}
                    # tqdm instead
                    log_str_short = f"ForceMAE={log_dict['forces_mae']:.2e}, EnergyMAE={log_dict['energy_mae']:.2e}, Loss={log_dict['loss']:.2e}"
                    # pbar.set_postfix_str(log_str_short)
                    pbar.set_description(get_pbar_desc(epoch_int))
                    pbar.write(f"Epoch={epoch_int}, step={i} : {log_str_short}")

                # if torch.isnan(log_dict['forces_mae']):
                # if log_dict['forces_mae'] == float("nan"): # nan, inf, -inf
                if math.isnan(log_dict["forces_mae"]):
                    isnan_cnt += 1
                    if isnan_cnt > len(self.train_loader) // 10:
                        raise ValueError("NaN detected in forces_mae. Exiting.")

                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split="train",
                    )

                if checkpoint_every != -1 and self.step % checkpoint_every == 0:
                    self.save(checkpoint_file="checkpoint.pt", training_state=True)
                    self.file_logger.info(f"Saved checkpoint at step {self.step}.")

                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0 or i == (len(self.train_loader) - 1):
                    if self.val_loader is not None:
                        if self.ema:
                            val_metrics = self.validate(
                                split="val",
                                disable_tqdm=disable_eval_tqdm,
                                use_ema=True,
                            )
                            self.update_best(
                                primary_metric,
                                val_metrics,
                                disable_eval_tqdm=disable_eval_tqdm,
                            )
                        else:
                            val_metrics = self.validate(
                                split="val",
                                disable_tqdm=disable_eval_tqdm,
                                use_ema=False,
                            )
                            self.update_best(
                                primary_metric,
                                val_metrics,
                                disable_eval_tqdm=disable_eval_tqdm,
                            )

                        if self.is_hpo:
                            self.hpo_update(
                                self.epoch,
                                self.step,
                                self.metrics,
                                val_metrics,
                            )

                    if self.config["task"].get("eval_relaxations", False):
                        if "relax_dataset" not in self.config["task"]:
                            logging.warning(
                                "Cannot evaluate relaxations, relax_dataset not specified"
                            )
                        else:
                            self.run_relaxations()

                    self.model.train()

                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    if self.grad_accumulation_steps != 1:
                        if self.step % self.grad_accumulation_steps == 0:
                            self.scheduler.step()
                    else:
                        self.scheduler.step()

                # end of step
                pbar.update(1)

            # end of epoch

            if checkpoint_every == -1:
                self.save(checkpoint_file="checkpoint.pt", training_state=True)

        # end of training
        # from datetime import datetime
        # self.file_logger.info(f'Finished training at time {datetime.now().time()}.')

        self.file_logger.info(f"Finished training at time {time.ctime()}.")

        pbar.close()
        self.train_dataset.close_db()
        if self.config.get("val_dataset", False):
            self.val_dataset.close_db()
        if self.config.get("test_dataset", False):
            self.test_dataset.close_db()

    def _forward(self, batch_list, fpreuse=False):
        # forward pass.
        if self.config["model_attributes"].get("regress_forces", True) or self.config[
            "model_attributes"
        ].get("use_auxiliary_task", False):
            if fpreuse:
                out_energy, out_forces, fp, info = self.model(batch_list, fixedpoint=self.fixedpoint, return_fixedpoint=True)
                self.fixedpoint = fp
            else:
                out_energy, out_forces, info = self.model(batch_list)
        else:
            if fpreuse:
                out_energy, fp, info = self.model(batch_list, fixedpoint=self.fixedpoint, return_fixedpoint=True)
                self.fixedpoint = fp
            else:
                out_energy, info = self.model(batch_list)

        if out_energy.shape[-1] == 1:
            out_energy = out_energy.view(-1)

        out = {
            "energy": out_energy,
            "info": info,
        }

        if self.config["model_attributes"].get("regress_forces", True) or self.config[
            "model_attributes"
        ].get("use_auxiliary_task", False):
            out["forces"] = out_forces

        return out

    def _compute_loss(self, out, batch_list, step=None):
        loss = []

        # Energy loss.
        energy_target = torch.cat(
            [batch.y.to(self.device) for batch in batch_list], dim=0
        )
        if self.normalizer.get("normalize_labels", False):
            energy_target = self.normalizers["target"].norm(energy_target)
        energy_mult = self.config["optim"].get("energy_coefficient", 1)
        loss.append(energy_mult * self.loss_fn["energy"](out["energy"], energy_target))

        if self.logger is not None:
            self.logger.log(
                {
                    "energy_pred_mean": out["energy"].mean().item(),
                    "energy_pred_std": out["energy"].std().item(),
                    "energy_pred_min": out["energy"].min().item(),
                    "energy_pred_max": out["energy"].max().item(),
                    "energy_target_mean": energy_target.mean().item(),
                    "energy_target_std": energy_target.std().item(),
                    "energy_target_min": energy_target.min().item(),
                    "energy_target_max": energy_target.max().item(),
                    "scaled_energy_loss": loss[-1].item(),
                },
                step=self.step,
                split="train",
            )

        # self.normalizer.get("normalize_labels", False) == True
        # self.loss_fn["energy"] == L1Loss()
        # self.config["task"].get("tag_specific_weights", []) == []
        # self.loss_fn["force"] == L2MAELoss()
        # self.config["optim"]["loss_force"] == l2mae
        # self.config["task"].get("train_on_free_atoms", False) == True
        # self.config["optim"].get("force_coefficient", 1) == 100

        # Force loss.
        if self.config["model_attributes"].get("regress_forces", True) or self.config[
            "model_attributes"
        ].get("use_auxiliary_task", False):
            force_target = torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            )
            if self.normalizer.get("normalize_labels", False):
                force_target = self.normalizers["grad_target"].norm(force_target)

            tag_specific_weights = self.config["task"].get("tag_specific_weights", [])
            if tag_specific_weights != []:
                # handle tag specific weights as introduced in forcenet
                assert len(tag_specific_weights) == 3

                batch_tags = torch.cat(
                    [batch.tags.float().to(self.device) for batch in batch_list],
                    dim=0,
                )
                weight = torch.zeros_like(batch_tags)
                weight[batch_tags == 0] = tag_specific_weights[0]
                weight[batch_tags == 1] = tag_specific_weights[1]
                weight[batch_tags == 2] = tag_specific_weights[2]

                loss_force_list = torch.abs(out["forces"] - force_target)
                train_loss_force_unnormalized = torch.sum(
                    loss_force_list * weight.view(-1, 1)
                )
                train_loss_force_normalizer = 3.0 * weight.sum()

                # add up normalizer to obtain global normalizer
                distutils.all_reduce(train_loss_force_normalizer)

                # perform loss normalization before backprop
                train_loss_force_normalized = train_loss_force_unnormalized * (
                    distutils.get_world_size() / train_loss_force_normalizer
                )
                loss.append(train_loss_force_normalized)

            else:
                # Force coefficient = 30 has been working well for us.
                force_mult = self.config["optim"].get("force_coefficient", 30)
                if self.config["task"].get("train_on_free_atoms", False):
                    fixed = torch.cat(
                        [batch.fixed.to(self.device) for batch in batch_list]
                    )
                    mask = fixed == 0
                    if self.config["optim"]["loss_force"].startswith("atomwise"):
                        force_mult = self.config["optim"].get("force_coefficient", 1)
                        natoms = torch.cat(
                            [batch.natoms.to(self.device) for batch in batch_list]
                        )
                        natoms = torch.repeat_interleave(natoms, natoms)
                        force_loss = force_mult * self.loss_fn["force"](
                            out["forces"][mask],
                            force_target[mask],
                            natoms=natoms[mask],
                            batch_size=batch_list[0].natoms.shape[0],
                        )
                        loss.append(force_loss)
                    else:
                        # ------------ Default ------------
                        loss.append(
                            force_mult
                            * self.loss_fn["force"](
                                out["forces"][mask], force_target[mask]
                            )
                        )
                else:
                    loss.append(
                        force_mult * self.loss_fn["force"](out["forces"], force_target)
                    )
            # computed forces
            if self.logger is not None:
                self.logger.log(
                    {
                        "force_pred_mean": out["forces"].mean().item(),
                        "force_pred_std": out["forces"].std().item(),
                        "force_pred_min": out["forces"].min().item(),
                        "force_pred_max": out["forces"].max().item(),
                        "force_target_mean": force_target.mean().item(),
                        "force_target_std": force_target.std().item(),
                        "force_target_min": force_target.min().item(),
                        "force_target_max": force_target.max().item(),
                        "scaled_force_loss": loss[-1].item(),
                    },
                    step=self.step,
                    split="train",
                )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")

        loss = sum(loss)
        return loss

    def _compute_metrics(
        self, out, batch_list, evaluator, metrics={}, return_results=False
    ):
        natoms = torch.cat(
            [batch.natoms.to(self.device) for batch in batch_list], dim=0
        )

        target = {
            "energy": torch.cat(
                [batch.y.to(self.device) for batch in batch_list], dim=0
            ),
            "forces": torch.cat(
                [batch.force.to(self.device) for batch in batch_list], dim=0
            ),
            "natoms": natoms,
        }

        out["natoms"] = natoms

        if self.config["task"].get("eval_on_free_atoms", True):
            fixed = torch.cat([batch.fixed.to(self.device) for batch in batch_list])
            mask = fixed == 0
            out["forces"] = out["forces"][mask]
            target["forces"] = target["forces"][mask]

            s_idx = 0
            natoms_free = []
            for natoms in target["natoms"]:
                natoms_free.append(torch.sum(mask[s_idx : s_idx + natoms]).item())
                s_idx += natoms
            target["natoms"] = torch.LongTensor(natoms_free).to(self.device)
            out["natoms"] = torch.LongTensor(natoms_free).to(self.device)

        if self.normalizer.get("normalize_labels", False):
            out["energy"] = self.normalizers["target"].denorm(out["energy"])
            out["forces"] = self.normalizers["grad_target"].denorm(out["forces"])

        if return_results:
            metrics, results = evaluator.eval(
                out, target, prev_metrics=metrics, return_results=return_results
            )
            return metrics, results
        else:
            metrics = evaluator.eval(out, target, prev_metrics=metrics)
            return metrics

    def run_relaxations(self, split="val"):
        self.file_logger.info("Running ML-relaxations")
        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator_is2rs, metrics_is2rs = Evaluator(task="is2rs"), {}
        evaluator_is2re, metrics_is2re = Evaluator(task="is2re"), {}

        if hasattr(self.relax_dataset[0], "pos_relaxed") and hasattr(
            self.relax_dataset[0], "y_relaxed"
        ):
            split = "val"
        else:
            split = "test"

        # load IS2RS pos predictions
        pred_pos_dict = None
        if self.config["task"]["relax_opt"].get("pred_pos_path", None):
            pred_pos_dict = torch.load(
                self.config["task"]["relax_opt"]["pred_pos_path"]
            )

        ids = []
        relaxed_positions = []
        chunk_idx = []
        for i, batch in tqdm(
            enumerate(self.relax_loader), total=len(self.relax_loader)
        ):
            if i >= self.config["task"].get("num_relaxation_batches", 1e9):
                break

            # If all traj files already exist, then skip this batch
            if check_traj_files(
                batch, self.config["task"]["relax_opt"].get("traj_dir", None)
            ):
                self.file_logger.info(f"Skipping batch: {batch[0].sid.tolist()}")
                continue

            # Initailize pos with IS2RS direct prediction
            if pred_pos_dict is not None:
                sid_list = batch[0].sid.tolist()
                pred_pos_list = []
                for sid in sid_list:
                    pred_pos_list.append(pred_pos_dict[str(sid)])
                pred_pos = torch.cat(pred_pos_list, dim=0)
                batch[0].pos = pred_pos

            relaxed_batch = ml_relax(
                batch=batch,
                model=self,
                steps=self.config["task"].get("relaxation_steps", 200),
                fmax=self.config["task"].get("relaxation_fmax", 0.0),
                relax_opt=self.config["task"]["relax_opt"],
                save_full_traj=True,
                device=self.device,
                transform=None,
            )

            if self.config["task"].get("write_pos", False):
                systemids = [str(i) for i in relaxed_batch.sid.tolist()]
                natoms = relaxed_batch.natoms.tolist()
                positions = torch.split(relaxed_batch.pos, natoms)
                batch_relaxed_positions = [pos.tolist() for pos in positions]

                relaxed_positions += batch_relaxed_positions
                chunk_idx += natoms
                ids += systemids

            if split == "val":
                mask = relaxed_batch.fixed == 0
                s_idx = 0
                natoms_free = []
                for natoms in relaxed_batch.natoms:
                    natoms_free.append(torch.sum(mask[s_idx : s_idx + natoms]).item())
                    s_idx += natoms

                target = {
                    "energy": relaxed_batch.y_relaxed,
                    "positions": relaxed_batch.pos_relaxed[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                prediction = {
                    "energy": relaxed_batch.y,
                    "positions": relaxed_batch.pos[mask],
                    "cell": relaxed_batch.cell,
                    "pbc": torch.tensor([True, True, True]),
                    "natoms": torch.LongTensor(natoms_free),
                }

                metrics_is2rs = evaluator_is2rs.eval(
                    prediction,
                    target,
                    metrics_is2rs,
                )
                metrics_is2re = evaluator_is2re.eval(
                    {"energy": prediction["energy"]},
                    {"energy": target["energy"]},
                    metrics_is2re,
                )

                # Log metrics.
                # log_dict = {k: metrics_is2re[k]["metric"] for k in metrics_is2re}
                # log_dict.update({k: metrics_is2rs[k]["metric"] for k in metrics_is2rs})
                if (
                    (i + 1) % self.config["cmd"]["print_every"] == 0
                    or i == 0
                    or i == (len(self.relax_loader) - 1)
                ):
                    distutils.synchronize()
                    log_dict = {}
                    for task in ["is2rs", "is2re"]:
                        metrics = eval(f"metrics_{task}")
                        aggregated_metrics = {}
                        for k in metrics:
                            aggregated_metrics[k] = {
                                "total": distutils.all_reduce(
                                    metrics[k]["total"],
                                    average=False,
                                    device=self.device,
                                ),
                                "numel": distutils.all_reduce(
                                    metrics[k]["numel"],
                                    average=False,
                                    device=self.device,
                                ),
                            }
                            aggregated_metrics[k]["metric"] = (
                                aggregated_metrics[k]["total"]
                                / aggregated_metrics[k]["numel"]
                            )
                            log_dict[k] = aggregated_metrics[k]["metric"]
                    log_str = ["{}: {:.2e}".format(k, v) for k, v in log_dict.items()]
                    log_str = ", ".join(log_str)
                    self.file_logger.info(
                        "[{}/{}] {}".format(i, len(self.relax_loader), log_str)
                    )

        if self.config["task"].get("write_pos", False):
            rank = distutils.get_rank()
            pos_filename = os.path.join(
                self.config["cmd"]["results_dir"], f"relaxed_pos_{rank}.npz"
            )
            np.savez_compressed(
                pos_filename,
                ids=ids,
                pos=np.array(relaxed_positions, dtype=object),
                chunk_idx=chunk_idx,
            )

            distutils.synchronize()
            if distutils.is_master():
                gather_results = defaultdict(list)
                full_path = os.path.join(
                    self.config["cmd"]["results_dir"],
                    "relaxed_positions.npz",
                )

                for i in range(distutils.get_world_size()):
                    rank_path = os.path.join(
                        self.config["cmd"]["results_dir"],
                        f"relaxed_pos_{i}.npz",
                    )
                    rank_results = np.load(rank_path, allow_pickle=True)
                    gather_results["ids"].extend(rank_results["ids"])
                    gather_results["pos"].extend(rank_results["pos"])
                    gather_results["chunk_idx"].extend(rank_results["chunk_idx"])
                    os.remove(rank_path)

                # Because of how distributed sampler works, some system ids
                # might be repeated to make no. of samples even across GPUs.
                _, idx = np.unique(gather_results["ids"], return_index=True)
                gather_results["ids"] = np.array(gather_results["ids"])[idx]
                gather_results["pos"] = np.concatenate(
                    np.array(gather_results["pos"])[idx]
                )
                gather_results["chunk_idx"] = np.cumsum(
                    np.array(gather_results["chunk_idx"])[idx]
                )[
                    :-1
                ]  # np.split does not need last idx, assumes n-1:end

                self.file_logger.info(f"Writing results to {full_path}")
                np.savez_compressed(full_path, **gather_results)

        if split == "val":
            for task in ["is2rs", "is2re"]:
                metrics = eval(f"metrics_{task}")
                aggregated_metrics = {}
                for k in metrics:
                    aggregated_metrics[k] = {
                        "total": distutils.all_reduce(
                            metrics[k]["total"],
                            average=False,
                            device=self.device,
                        ),
                        "numel": distutils.all_reduce(
                            metrics[k]["numel"],
                            average=False,
                            device=self.device,
                        ),
                    }
                    aggregated_metrics[k]["metric"] = (
                        aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
                    )
                metrics = aggregated_metrics

                # Make plots.
                log_dict = {f"{task}_{k}": metrics[k]["metric"] for k in metrics}
                if self.logger is not None:
                    self.logger.log(
                        log_dict,
                        step=self.step,
                        split=split,
                    )

                if distutils.is_master():
                    self.file_logger.info(metrics)

        if self.ema:
            self.ema.restore()

    @torch.no_grad()
    def validate(self, split="val", disable_tqdm=False, use_ema=False):
        self.file_logger.info(f"Evaluating on {split}.")

        if self.is_hpo:
            disable_tqdm = True

        if self.config["test_w_eval_mode"]:
            self.model.eval()
        if self.ema and use_ema:
            self.ema.store()
            self.ema.copy_to()

        evaluator, metrics = Evaluator(task=self.name), {}
        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        # max_iter = self.config.get("val_max_iter", -1)
        max_iter = self.config.get("val_max_iter", -1)
        if max_iter == -1:
            max_iter = len(loader)
        self.file_logger.info(f"Validate max steps: {max_iter}")

        def get_pbar_desc(step):
            return f"Validate (device {distutils.get_rank()}) Step={step}"

        start_time = time.perf_counter()

        for i, batch in tqdm(
            enumerate(loader),
            total=max_iter,
            position=rank,
            desc="Validate: device {}".format(rank),
            unit="steps",
            disable=disable_tqdm,
        ):
            if i >= max_iter:
                break
            with torch.cuda.amp.autocast(enabled=self.scaler is not None):
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)
            # DEQ logging
            if "nstep" in out["info"]:
                metrics = evaluator.update(
                    key="nstep",
                    stat=out["info"]["nstep"].mean().item(),
                    metrics=metrics,
                )
            # log fixed-point trajectory once per evaluation
            if i == 0:
                if "abs_trace" in out["info"].keys():
                    logging_utils_deq.log_fixed_point_error(
                        out["info"],
                        step=self.step,
                        split="val",
                    )

        # if distutils.is_master():
        time_total = time.perf_counter() - start_time

        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        metrics = aggregated_metrics

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict["time_total"] = time_total
        log_dict.update({"epoch": self.epoch})
        log_str = ["{}: {:.4f}".format(k, v) for k, v in log_dict.items()]
        log_str = ", ".join(log_str)
        log_str = "[{}] ".format(split) + log_str
        self.file_logger.info(log_str)

        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )
        else:
            self.file_logger.info(f"validate: self.logger is None. log_dict: {log_dict}")

        if self.ema and use_ema:
            self.ema.restore()

        return metrics
