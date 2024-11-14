# Run relxation, starting from configurations in the OC20 dataset
# Run the relaxation using the LBFGS optimizer for a maximum of 
# ... steps or until the maximum force on any atom is below ... eV/Å1
# The relaxation trajectory will be saved in the "data" directory as ...
# and logged to wandb.

# Usage:
# launchrelax preset=reg +use=deq +cfg=ap2
# launchrelax preset=reg +use=deq +cfg=ap2 fpreuse_test=False
# launchrelax preset=reg +use=deq +cfg=ap2 deq_kwargs_fpr.f_tol=1e-1
# launchrelax preset=reg +cfg=dd model.num_layers=12

# bash command to download model checkpoint checkpoints/pDEQsap2reg/best_checkpoint.pt
# from andreasburger@tacozoid.accelerationconsortium.ai
# mkdir -p checkpoints/pDEQsap2reg
# scp andreasburger@tacozoid.accelerationconsortium.ai:/home/andreasburger/equilibrium-forcefields/checkpoints/pDEQsap2reg/best_checkpoint.pt ./checkpoints/pDEQsap2reg/best_checkpoint.pt
# mkdir -p checkpoints/pEsddnumlayers12reg
# scp andreasburger@tacozoid.accelerationconsortium.ai:/home/andreasburger/equilibrium-forcefields/checkpoints/pEsddnumlayers12reg/best_checkpoint.pt ./checkpoints/pEsddnumlayers12reg/best_checkpoint.pt

import os
import wandb
import argparse
import torch
import copy
import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf
import time

from deq2ff.oc20runner import get_OC20runner
from ocpmodels.common.relaxation.ase_utils import OCPCalculator
from deq2ff.logging_utils import init_wandb, fix_args_set_name

from ocpmodels.common.utils import radius_graph_pbc

from ase.build import fcc100, molecule
from ase.constraints import FixAtoms
from ase.optimize import LBFGS
from ase.io.trajectory import Trajectory

from ocpmodels.common.relaxation.ase_utils import batch_to_atoms

def _forward_otf_graph(data, max_radius, max_neighbors):
    edge_index, cell_offsets, neighbors = radius_graph_pbc(
        data, max_radius, max_neighbors
    )
    data.edge_index = edge_index
    data.cell_offsets = cell_offsets
    data.neighbors = neighbors
    return data

@hydra.main(
    config_name="oc20", config_path="../equiformer_v2/config", version_base="1.3"
)
def hydra_wrapper_relax(args: DictConfig) -> None:

    args.optim.batch_size = 1
    args.logger.project = "relaxOC20"
    
    # sets wandb_run_name
    args = fix_args_set_name(args)

    # turn args into dictionary for compatibility with ocpmodels
    # args: omegaconf.dictconfig.DictConfig -> dict
    args = OmegaConf.to_container(args, resolve=True)
    
    # Hacky way to make sure we have the exactly same config 
    # for the relaxation as for training.
    # We load the trainer, get the relevant info, delete the trainer,
    # and then use that info to create the runner for relaxation.
    argsref = copy.deepcopy(args)
    argsref['wandb'] = False
    argsref['logger'] = "dummy"
    runner = get_OC20runner(argsref)
    reference = runner.trainer
    del reference.model
    
    # "checkpoints/pDEQsap2reg/best_checkpoint.pt"
    checkpoint_path = copy.deepcopy(reference.config["cmd"]["checkpoint_dir"])
    checkpoint_path += "/best_checkpoint.pt"

    train_loader = runner.trainer.train_loader
    del runner, reference
    print("train_loader", train_loader, len(train_loader))
    
    for i, sample in enumerate(train_loader):
        if i >= args['relax']['n_samples']:
            break
        
        # Trainer will pop keys out of args
        argsstep = copy.deepcopy(args)

        # TODO: not sure if to run: 
        # with forces_v2 or don't specify
        # with config_yml or don't specify
        max_radius = argsstep['model']['max_radius']
        max_neighbors = argsstep['model']['max_neighbors']
        argsstep['wandb_run_name'] = argsstep['wandb_run_name'] + f" s{i}"
        ocp_calculator = OCPCalculator(
            # config_yml="equiformer_v2/config/oc20.yaml",
            config_yml=argsstep,
            checkpoint=checkpoint_path,
            trainer="forces_v2", # forces, forces_v2
            cutoff=max_radius,
            max_neighbors=max_neighbors,
            device=0,
            identifier=argsstep['wandb_run_name']
        )
        ocp_calculator.trainer.fpreuse_test = True

        if args['relax']['system'] == "oc20":
            # get sample from OC20 dataset
            data = _forward_otf_graph(sample[0], max_radius, max_neighbors)
            atoms = batch_to_atoms(data)
            print("atoms", atoms)
            adslab = atoms[0]
            
        else:
            raise NotImplementedError(f"System {args['relax']['system']} not implemented")
            # Create an adslab system using ASE. 
            # Cu(100) surface with a CH3O adsorbate:
            args['relax']['system'] = "Cu100_CH3O"
            
            # Create the Cu(100) surface
            adslab = fcc100("Cu", size=(3, 3, 3))
        
            # Add the CH3O adsorbate
            adsorbate = molecule("CH3O")
            adslab.extend(adsorbate)
            
            # adslab: Atoms
            # Symbols('Cu27COH3')
            # PBC array([ True,  True, False])
            # Cell([7.65796644025031, 7.65796644025031, 0.0])

            # Fix the bottom two layers of the slab
            constraint = FixAtoms(
                indices=[atom.index for atom in adslab if atom.position[2] < 7]
            )
            adslab.set_constraint(constraint)
        
        
        print("adslab.cell", adslab.cell)
        print("adslab.pbc", adslab.pbc)
        # atom positions
        for atom in adslab:
            print(atom.symbol, atom.position)

        # Set the calculator
        adslab.set_calculator(ocp_calculator)

        # Create a directory to store the trajectory
        logdir = "relax_logs"
        os.makedirs(logdir, exist_ok=True)

        # Set up the optimizer
        dyn = LBFGS(
            adslab, 
            trajectory=os.path.join(
                logdir, f"{args['relax']['system']}.traj"
            )
        )

        # Run the relaxation
        print("Running relaxation...")
        print("-"*50)
        start = time.time()
        dyn.run(
            fmax=args['relax']['fmax'], 
            steps=args['relax']['steps']
        )
        time_taken = time.time() - start
        
        wandb.log({"time_taken": time_taken, "relax_nsteps": dyn.nsteps, "time_per_step": time_taken / dyn.nsteps})
        
        wandb.finish()
        
        # delete open loggers
        # ocp_calculator.trainer.logger.close() # WandB / Tensorboard
        ocp_calculator.trainer.file_logger.close()
        del ocp_calculator
    
    print("Relaxations completed ✅")
    
if __name__ == "__main__":
    hydra_wrapper_relax()