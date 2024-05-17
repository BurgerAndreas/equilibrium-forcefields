#!/bin/bash

# replace `launchrun` with `sbatch scripts/amd_launcher.slrm train_deq_md_v2.py` or `sbatch scripts/slurm_launcher.slrm train_deq_md_v2.py`

md17=(aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)
md22=(AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose)

layers=(1 2 3 4 5 6 7 8)

for l in ${layers[@]}; do
  launchrun model.num_layers=$l 
done

# # launchrun model.num_layers=1 wandb_tags=["depth"]
# launchrun model.num_layers=2 wandb_tags=["depth"]
# launchrun model.num_layers=3 wandb_tags=["depth"]
# # launchrun model.num_layers=4 wandb_tags=["depth"]
# launchrun model.num_layers=5 wandb_tags=["depth"]
# launchrun model.num_layers=6 wandb_tags=["depth"]
# launchrun model.num_layers=7 wandb_tags=["depth"]
# launchrun model.num_layers=8 wandb_tags=["depth"]

# seed=2
# launchrun model.num_layers=1 seed=2 wandb_tags=["depth"]
launchrun model.num_layers=2 seed=2 wandb_tags=["depth"]
launchrun model.num_layers=3 seed=2 wandb_tags=["depth"]
# launchrun model.num_layers=4 seed=2 wandb_tags=["depth"]
launchrun model.num_layers=5 seed=2 wandb_tags=["depth"]
launchrun model.num_layers=6 seed=2 wandb_tags=["depth"]
launchrun model.num_layers=7 seed=2 wandb_tags=["depth"]
launchrun model.num_layers=8 seed=2 wandb_tags=["depth"]

# seed=3
# launchrun model.num_layers=1 seed=3 wandb_tags=["depth"]
launchrun model.num_layers=2 seed=3 wandb_tags=["depth"]
launchrun model.num_layers=3 seed=3 wandb_tags=["depth"]
# launchrun model.num_layers=4 seed=3 wandb_tags=["depth"]
launchrun model.num_layers=5 seed=3 wandb_tags=["depth"]
launchrun model.num_layers=6 seed=3 wandb_tags=["depth"]
launchrun model.num_layers=7 seed=3 wandb_tags=["depth"]
launchrun model.num_layers=8 seed=3 wandb_tags=["depth"]

# ethanol
# REMARK: submitted to taco
# launchrun model.num_layers=1 target=ethanol wandb_tags=["depth"]
launchrun model.num_layers=2 target=ethanol wandb_tags=["depth"]
launchrun model.num_layers=3 target=ethanol wandb_tags=["depth"]
# launchrun model.num_layers=4 target=ethanol wandb_tags=["depth"]
launchrun model.num_layers=5 target=ethanol wandb_tags=["depth"]
launchrun model.num_layers=6 target=ethanol wandb_tags=["depth"]
launchrun model.num_layers=7 target=ethanol wandb_tags=["depth"]
launchrun model.num_layers=8 target=ethanol wandb_tags=["depth"]

