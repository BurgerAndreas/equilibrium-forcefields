#!/bin/bash

# replace `launchrun` with `sbatch scripts/amd_launcher.slrm train_deq_md_v2.py` or `sbatch scripts/slurm_launcher.slrm train_deq_md_v2.py`

# run with `source multiruns/md22_deq.sh`

md17=(aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)
md22=(AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose)

for mol in ${md22[@]}; do
    launchrun +use=deq +cfg=bp wandb_tags=["md22"] dname=md22 target=$mol
    # launchrun model.num_layers=1 +cfg=dd wandb_tags=["md17"] target=$mol
    # launchrun model.num_layers=4 +cfg=dd  wandb_tags=["md17"] target=$mol
    # launchrun model.num_layers=8 +cfg=dd  wandb_tags=["md17"] target=$mol
done