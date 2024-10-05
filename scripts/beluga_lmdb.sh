#!/bin/bash
# '-t 3-0' for three days
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpunodes
# --gpus-per-node=v100:1 or  --gres=gpu[[:type]:number]
#SBATCH --gres=gpu:1 # any
# constraint="RTX_A4500|GTX_1080_Ti" 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8 # number of cores
#SBATCH --ntasks=1
# --mem-per-cpu (memory per core) or --mem (memory per node)
#SBATCH --mem=16G
#SBATCH --job-name=equiformer 
#SBATCH --output=outslurm/slurm-%j.txt 
#SBATCH --error=outslurm/slurm-%j.txt

# specify list via features: --gres=gpu:1 --constraint="RTX_A4500|GTX_1080_Ti"
# # srun --partition gpunodes -c 4 --mem=8G --gres=gpu:1 --constraint="RTX_A4500|GTX_1080_Ti" -t 60 --pty bash --login

# # slurm_report -g
# # Allocated/Idle/Other/Total

# # scontrol show nodes

module load StdEnv/2020 cudacore/.11.7.0 cuda/11.7 cudnn/8.9.5.29

echo `date`: Job $SLURM_JOB_ID is allocated resources.

# /home/aburger/equilibrium-forcefields
HOME_DIR=/home/aburger

export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/ocp/scripts/download_data.py

# hand over all arguments to the script
echo "Submitting ${SCRIPTDIR}"

${HOME_DIR}/miniforge3/envs/deq/bin/python ${SCRIPTDIR} --task s2ef --split "200k" --num-workers 8 --ref-energy
--data-path /project/def-aspuru/aburger/deq2ff/data
{HOME_DIR}/miniforge3/envs/deq/bin/python ${SCRIPTDIR} --task s2ef --split "2M" --num-workers 8 --ref-energy
--data-path /project/def-aspuru/aburger/deq2ff/data

# sbatch scripts/beluga_lmdb.sh