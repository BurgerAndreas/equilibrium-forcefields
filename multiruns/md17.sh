#!/bin/bash

# replace `launchrun` with `sbatch scripts/amd_launcher.slrm train_deq_md_v2.py` or `sbatch scripts/slurm_launcher.slrm train_deq_md_v2.py`

# run with `bash multiruns/md17.sh`

md17=(aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)
md22=(AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose)

for mol in ${md17[@]}; do
    launchrun +use=deq +cfg=fpc_of wandb_tags=["md17"] target=$mol
    launchrun model.num_layers=1 wandb_tags=["md17"] target=$mol
    launchrun model.num_layers=4 wandb_tags=["md17"] target=$mol
done

# Baseline Equiformer2 with one layer
# launchrun model.num_layers=1 wandb_tags=["md17"] target=aspirin
# launchrun model.num_layers=1 wandb_tags=["md17"] target=benzene
# launchrun model.num_layers=1 wandb_tags=["md17"] target=ethanol
# launchrun model.num_layers=1 wandb_tags=["md17"] target=malonaldehyde
# launchrun model.num_layers=1 wandb_tags=["md17"] target=naphthalene
# launchrun model.num_layers=1 wandb_tags=["md17"] target=salicylic_acid
# launchrun model.num_layers=1 wandb_tags=["md17"] target=toluene
# launchrun model.num_layers=1 wandb_tags=["md17"] target=uracil


# Baseline Equiformer2 with four layers
# launchrun model.num_layers=4 wandb_tags=["md17"] target=aspirin
# launchrun model.num_layers=4 wandb_tags=["md17"] target=benzene
# launchrun model.num_layers=4 wandb_tags=["md17"] target=ethanol
# launchrun model.num_layers=4 wandb_tags=["md17"] target=malonaldehyde
# launchrun model.num_layers=4 wandb_tags=["md17"] target=naphthalene
# launchrun model.num_layers=4 wandb_tags=["md17"] target=salicylic_acid
# launchrun model.num_layers=4 wandb_tags=["md17"] target=toluene
# launchrun model.num_layers=4 wandb_tags=["md17"] target=uracil

# TODO: replace this with a loop for all molecules
# seed=2
# launchrun +use=deq +cfg=fpc_of seed=2
# launchrun model.num_layers=1 seed=2
# launchrun model.num_layers=4 seed=2

# seed=3
# launchrun +use=deq +cfg=fpc_of seed=3
# launchrun model.num_layers=1 seed=3
# launchrun model.num_layers=4 seed=3


########################################################################################
# With dropout
########################################################################################

# # DEQ 1 layer
# # launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 wandb_tags=["md17"] target=aspirin
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 wandb_tags=["md17"] target=benzene
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 wandb_tags=["md17"] target=ethanol
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 wandb_tags=["md17"] target=malonaldehyde
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 wandb_tags=["md17"] target=naphthalene
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 wandb_tags=["md17"] target=salicylic_acid
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 wandb_tags=["md17"] target=toluene
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 wandb_tags=["md17"] target=uracil

# # DEQ 2 layers
# # launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=aspirin
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=benzene
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=ethanol
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=malonaldehyde
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=naphthalene
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=salicylic_acid
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=toluene
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=uracil

# # Baseline Equiformer2 with one layer
# launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=aspirin
# launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=benzene
# launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=ethanol
# launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=malonaldehyde
# launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=naphthalene
# launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=salicylic_acid
# launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=toluene
# launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=uracil

# # Baseline Equiformer2 with four layers
# launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=aspirin
# launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=benzene
# launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=ethanol
# launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=malonaldehyde
# launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=naphthalene
# launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=salicylic_acid
# launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=toluene
# launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=uracil

# # Baseline Equiformer2 with eight layers
# launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=aspirin
# launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=benzene
# launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=ethanol
# launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=malonaldehyde
# launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=naphthalene
# launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=salicylic_acid
# launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=toluene
# launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=uracil

########################################################
# MD22
# AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose

# DEQ 2 layers
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=AT_AT_CG_CG
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=AT_AT
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=Ac_Ala3_NHMe
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=DHA
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=buckyball_catcher
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=dw_nanotube
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.drop_path_rate=0.05 wandb_tags=["md17"] target=stachyose

# Baseline Equiformer2 with one layer
launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=AT_AT_CG_CG
launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=AT_AT
launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=Ac_Ala3_NHMe
launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=DHA
launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=buckyball_catcher
launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=dw_nanotube
launchrun model.num_layers=1 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=stachyose

# Baseline Equiformer2 with four layers
launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=AT_AT_CG_CG
launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=AT_AT
launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=Ac_Ala3_NHMe
launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=DHA
launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=buckyball_catcher
launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=dw_nanotube
launchrun model.num_layers=4 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=stachyose

# Baseline Equiformer2 with eight layers
launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=AT_AT_CG_CG
launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=AT_AT
launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=Ac_Ala3_NHMe
launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=DHA
launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=buckyball_catcher
launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=dw_nanotube
launchrun model.num_layers=8 model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["md17"] target=stachyose