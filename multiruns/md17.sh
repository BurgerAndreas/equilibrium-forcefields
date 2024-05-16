#!/bin/bash

# replace `launchrun` with `sbatch scripts/amd_launcher.slrm train_deq_md_v2.py` or `sbatch scripts/slurm_launcher.slrm train_deq_md_v2.py`

# run with `bash multiruns/md17.sh`

md17=(aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)
md22=(AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose)

for mol in ${md17[@]}; do
  launchrun +use=deq +cfg=fpc_of target=$mol
#   launchrun model.num_layers=1 target=$mol
#   launchrun model.num_layers=4 target=$mol
done

# Baseline Equiformer2 with one layer
# launchrun model.num_layers=1 target=aspirin
# launchrun model.num_layers=1 target=benzene
# launchrun model.num_layers=1 target=ethanol
# launchrun model.num_layers=1 target=malonaldehyde

# launchrun model.num_layers=1 target=naphthalene
# launchrun model.num_layers=1 target=salicylic_acid
# launchrun model.num_layers=1 target=toluene
# launchrun model.num_layers=1 target=uracil


# Baseline Equiformer2 with four layers
# launchrun model.num_layers=4 target=aspirin
# launchrun model.num_layers=4 target=benzene
# launchrun model.num_layers=4 target=ethanol
# launchrun model.num_layers=4 target=malonaldehyde

# launchrun model.num_layers=4 target=naphthalene
# launchrun model.num_layers=4 target=salicylic_acid
# launchrun model.num_layers=4 target=toluene
# launchrun model.num_layers=4 target=uracil

# TODO: replace this with a loop for all molecules
# seed=2
# launchrun +use=deq +cfg=fpc_of seed=2
# launchrun model.num_layers=1 seed=2
# launchrun model.num_layers=4 seed=2

# seed=3
# launchrun +use=deq +cfg=fpc_of seed=3
# launchrun model.num_layers=1 seed=3
# launchrun model.num_layers=4 seed=3