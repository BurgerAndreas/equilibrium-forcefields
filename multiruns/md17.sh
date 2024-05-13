#!/bin/bash

# replace `run` with `sbatch scripts/amd_launcher.slrm train_deq_md_v2.py` or `sbatch scripts/slurm_launcher.slrm train_deq_md_v2.py`

md17=(aspirin benzene ethanol malonaldehyde naphtalene salicyclic_acid toluene uracil)
md22=(AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose)

for mol in ${md17[@]}; do
  run +use=deq +cfg=fpc_of target=$mol
#   run model.num_layers=1 target=$mol
#   run model.num_layers=4 target=$mol
done

# Baseline Equiformer2 with one layer
run model.num_layers=1 target=aspirin
run model.num_layers=1 target=benzene
run model.num_layers=1 target=ethanol
run model.num_layers=1 target=malonaldehyde

# run model.num_layers=1 target=naphtalene
# run model.num_layers=1 target=salicyclic_acid
# run model.num_layers=1 target=toluene
# run model.num_layers=1 target=uracil


# Baseline Equiformer2 with four layers
# run model.num_layers=4 target=aspirin
# run model.num_layers=4 target=benzene
# run model.num_layers=4 target=ethanol
# run model.num_layers=4 target=malonaldehyde

# run model.num_layers=4 target=naphtalene
# run model.num_layers=4 target=salicyclic_acid
# run model.num_layers=4 target=toluene
# run model.num_layers=4 target=uracil