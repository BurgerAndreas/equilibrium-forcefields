#!/bin/bash

md17=(aspirin benzene ethanol malonaldehyde naphtalene salicyclic_acid toluene uracil)
md22=(AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose)

for mol in ${md17[@]}; do
  sbatch scripts/amd_launcher.slrm train_deq_md_v2.py +use=deq +cfg=fpc_of target=$mol
#   sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=$mol
#   sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=$mol
done

# Baseline Equiformer2 with one layer
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=aspirin
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=benzene
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=ethanol
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=malonaldehyde

# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=naphtalene
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=salicyclic_acid
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=toluene
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=1 target=uracil


# Baseline Equiformer2 with four layers
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=aspirin
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=benzene
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=ethanol
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=malonaldehyde

# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=naphtalene
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=salicyclic_acid
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=toluene
# sbatch scripts/amd_launcher.slrm train_deq_md_v2.py model.num_layers=4 target=uracil