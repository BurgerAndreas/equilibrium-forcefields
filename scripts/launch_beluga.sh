#!/bin/bash
# '-t 3-0' for three days
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpunodes
# gres=gpu:<>:1 rtx_4090 rtx_a4000 rtx_a4500 rtx_a2000 rtx_a6000 gtx_1070 gtx_1080_ti rtx_2060 rtx_2070 rtx_2080
#SBATCH --gres=gpu:1 # any
# constraint="RTX_A4500|GTX_1080_Ti" 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8 # number of cores
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --job-name=equiformer 
#SBATCH --output=outslurm/slurm-%j.txt 
#SBATCH --error=outslurm/slurm-%j.txt

# specify list via features: --gres=gpu:1 --constraint="RTX_A4500|GTX_1080_Ti"
# # srun --partition gpunodes -c 4 --mem=8G --gres=gpu:1 --constraint="RTX_A4500|GTX_1080_Ti" -t 60 --pty bash --login

# # slurm_report -g
# # Allocated/Idle/Other/Total

# # scontrol show nodes

# # Example usage:
# # srun --partition gpunodes -c 4 --mem=8G --gres=gpu:rtx_a2000:1 -t 60 --pty bash --login

# module load python pytorch
# module load miniforge3 # miniconda3 miniforge3

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "Inside slurm_launcher.slrm ($0). received arguments: $@"

# /home/aburger/equilibrium-forcefields
HOME_DIR=/home/aburger

# export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/equilibrium-forcefields/train
export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/equiformer
if [[ $1 == *"test"* ]]; then
    echo "Found test in the filename. Changing the scriptdir to equilibrium-forcefields/tests"
    export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/tests
elif [[ $1 == *"deq"* ]]; then
    echo "Found deq in the filename. Changing the scriptdir to equilibrium-forcefields/scripts"
    export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/scripts
fi

# hand over all arguments to the script
echo "Submitting ${SCRIPTDIR}/$@"

${HOME_DIR}/miniforge3/envs/deq/bin/python ${SCRIPTDIR}/"$@"