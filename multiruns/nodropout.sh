#!/bin/bash


# md17=(aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)
# DEQ 2 layers all molecules 
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 wandb_tags=["md17"] target=aspirin
launchrun +use=deq +cfg=fpc_of model.num_layers=2 wandb_tags=["md17"] target=benzene
launchrun +use=deq +cfg=fpc_of model.num_layers=2 wandb_tags=["md17"] target=ethanol
launchrun +use=deq +cfg=fpc_of model.num_layers=2 wandb_tags=["md17"] target=malonaldehyde
launchrun +use=deq +cfg=fpc_of model.num_layers=2 wandb_tags=["md17"] target=naphthalene
launchrun +use=deq +cfg=fpc_of model.num_layers=2 wandb_tags=["md17"] target=salicylic_acid
launchrun +use=deq +cfg=fpc_of model.num_layers=2 wandb_tags=["md17"] target=toluene
launchrun +use=deq +cfg=fpc_of model.num_layers=2 wandb_tags=["md17"] target=uracil

# Equiformer 8 layers all molecules
# launchrun model.num_layers=8 wandb_tags=["md17"] target=aspirin
launchrun model.num_layers=8 wandb_tags=["md17"] target=benzene
launchrun model.num_layers=8 wandb_tags=["md17"] target=ethanol
launchrun model.num_layers=8 wandb_tags=["md17"] target=malonaldehyde
launchrun model.num_layers=8 wandb_tags=["md17"] target=naphthalene
launchrun model.num_layers=8 wandb_tags=["md17"] target=salicylic_acid
launchrun model.num_layers=8 wandb_tags=["md17"] target=toluene
launchrun model.num_layers=8 wandb_tags=["md17"] target=uracil
