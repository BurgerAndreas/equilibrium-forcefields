#!/bin/bash
#SBATCH --time=00:01:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name=test 
#SBATCH --output=slurmtest.txt 
#SBATCH --error=slurmtest.err

# module load python pytorch
module load miniforge3

echo " "
echo "Pytorch"

python -c "import torch; import torch; print('torch.__version__', torch.__version__); import torch_geometric; print('torch_geometric.__version__', torch_geometric.__version__)" 
python -c "print('torch.version.cuda', torch.version.cuda);"

# launch via: sbatch scripts/test_pytorch.slrm