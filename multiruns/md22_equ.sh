#!/bin/bash

# replace `launchrun` with `sbatch scripts/amd_launcher.slrm train_deq_md_v2.py` or `sbatch scripts/slurm_launcher.slrm train_deq_md_v2.py`

# run with `source multiruns/md17.sh`

md17=(aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)
md22=(AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose)

for mol in ${md22[@]}; do
    # launchrun +use=deq +cfg=fpc_of wandb_tags=["md17"] target=$mol
    launchrun model.num_layers=1 +cfg=dd wandb_tags=["md17"] target=$mol
    launchrun model.num_layers=4 +cfg=dd  wandb_tags=["md17"] target=$mol
    launchrun model.num_layers=8 +cfg=dd  wandb_tags=["md17"] target=$mol
done


# # Baseline Equiformer2 with one layer
# launchrun model.num_layers=1 +cfg=dd wandb_tags=["md223"] dname=md22 target=AT_AT_CG_CG
# launchrun model.num_layers=1 +cfg=dd wandb_tags=["md223"] dname=md22 target=AT_AT
# launchrun model.num_layers=1 +cfg=dd wandb_tags=["md223"] dname=md22 target=Ac_Ala3_NHMe
# launchrun model.num_layers=1 +cfg=dd wandb_tags=["md223"] dname=md22 target=DHA
# launchrun model.num_layers=1 +cfg=dd wandb_tags=["md223"] dname=md22 target=buckyball_catcher
# launchrun model.num_layers=1 +cfg=dd wandb_tags=["md223"] dname=md22 target=dw_nanotube # batch_size=2
# launchrun model.num_layers=1 +cfg=dd wandb_tags=["md223"] dname=md22 target=stachyose

# # Baseline Equiformer2 with four layers
# # dw_nanotube might require batch_size=2
# launchrun model.num_layers=4 +cfg=dd wandb_tags=["md223"] dname=md22 target=AT_AT_CG_CG
# launchrun model.num_layers=4 +cfg=dd wandb_tags=["md223"] dname=md22 target=AT_AT
# launchrun model.num_layers=4 +cfg=dd wandb_tags=["md223"] dname=md22 target=Ac_Ala3_NHMe
# launchrun model.num_layers=4 +cfg=dd wandb_tags=["md223"] dname=md22 target=DHA
# launchrun model.num_layers=4 +cfg=dd wandb_tags=["md223"] dname=md22 target=buckyball_catcher
# launchrun model.num_layers=4 +cfg=dd wandb_tags=["md223"] dname=md22 target=dw_nanotube batch_size=1 # lr=3e-4
# launchrun model.num_layers=4 +cfg=dd wandb_tags=["md223"] dname=md22 target=stachyose

# # Baseline Equiformer2 with eight layers
# # dw_nanotube might require batch_size=1
# launchrun model.num_layers=8 +cfg=dd wandb_tags=["md223"] dname=md22 target=AT_AT_CG_CG
# launchrun model.num_layers=8 +cfg=dd wandb_tags=["md223"] dname=md22 target=AT_AT
# launchrun model.num_layers=8 +cfg=dd wandb_tags=["md223"] dname=md22 target=Ac_Ala3_NHMe
# launchrun model.num_layers=8 +cfg=dd wandb_tags=["md223"] dname=md22 target=DHA
# launchrun model.num_layers=8 +cfg=dd wandb_tags=["md223"] dname=md22 target=buckyball_catcher
# launchrun model.num_layers=8 +cfg=dd wandb_tags=["md223"] dname=md22 target=dw_nanotube batch_size=1
# launchrun model.num_layers=8 +cfg=dd wandb_tags=["md223"] dname=md22 target=stachyose