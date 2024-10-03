# Installation

Get miniforge (mamba)
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc
~/miniforge3/bin/mamba
source ~/.bashrc
```

Get a terminal based editor
```bash
curl https://getmic.ro | bash
# add to bashrc
echo 'alias micro="~/micro"' >> ~/.bashrc
source ~/.bashrc
```

acces to cluster via ssh
```bash
ssh-copy-id <username>@remote_host
```

authenticate with github
```bash
# generate ssh key
ssh-keygen -t ed25519 -C "your_email@example.com"
# add ssh key to ssh-agent
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
# add ssh key permanently
micro ~/.ssh/config
# paste this
Host github.com
    IdentityFile ~/.ssh/id_ed25519
# stop paste
# add ssh key to github
cat ~/.ssh/id_ed25519.pub
# https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account
```

```bash
git clone git@github.com:BurgerAndreas/equilibrium-forcefields.git
cd equilibrium-forcefields
mamba create -n deq10 python=3.10
mamba activate deq

# get the Open Catalyst Project (required for Equiformerv2)
# outdated: git clone git@github.com:Open-Catalyst-Project/ocp.git
# https://github.com/FAIR-Chem/fairchem/blob/v0.1.0
cd ocp
# no longer necessary
# git checkout v0.1.0
# mamba env update --name deq --file env.common.yml --prune
pip install -e .
pre-commit install
# extra packages
pip install setuptools==57.4.0
pip install demjson 
# pip install demjson3
pip install lmdb==1.1.1
pip install "ray[tune]"
pip install submitit
cd ..

# Get the OCP data (optional)
cd ocp
# Structure to Energy and Forces (S2EF) task
# "2M": 3.4GB (17GB uncompressed)
# https://github.com/Open-Catalyst-Project/ocp/blob/main/DATASET.md
# python scripts/download_data.py --task s2ef --split "2M" --num-workers 8 --ref-energy 
python scripts/download_data.py --task s2ef --split "200k" --num-workers 8 --ref-energy 
python scripts/download_data.py --task s2ef --split "val_id" --num-workers 8 --ref-energy 
# More data splits:
# python scripts/download_data.py --task is2re
# python scripts/download_data.py --task s2ef --split test
cd ..

# install equiformer_v2
# git clone git@github.com:atomicarchitects/equiformer_v2.git
cd equiformer_v2
pip install -e .
cd ..

# install equiformer
# git clone git@github.com:atomicarchitects/equiformer.git
cd equiformer
pip install -e .
# mamba env update --name deq --file env/env_equiformer.yml --prune
cd ..

# link OCP dataset for equiformer (optional)
cd equiformer/datasets
mkdir oc20
cd oc20
ln -s ../../../ocp/data/is2re is2re
cd ../../..

# 1) Nvidia CUDA
# check your cuda version 
nvidia-smi
# https://github.com/pyg-team/pytorch_geometric/issues/999#issuecomment-606565132
# ls /pkgs/cuda- 
# cuda-12.1/
pip uninstall torch torchvision torch-cluster torch-geometric torch-scatter torch-sparse torch-spline-conv -y
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
# OCP requires torch-geometric<=2.0.4
pip install torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
# pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 --index-url https://data.pyg.org/whl/torch-1.11.0+cu115.html
# pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu118
# pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

# double check cuda version
# ls ~/miniforge3/envs/deq/lib/python3.9/site-packages/
pip freeze | grep nvidia
mamba list -n deq | grep cu

# 2) AMD ROCm
rocm-smi
apt show rocm-libs -a
# Package: rocm-libs Version: 5.3.0.50300-63~22.04
pip uninstall torch torchvision torch-cluster torch-geometric torch-scatter torch-sparse torch-spline-conv -y
pip3 install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/rocm5.3
# https://github.com/Looong01/pyg-rocm-build/releases
# V2
wget https://github.com/Looong01/pyg-rocm-build/releases/download/3/torch-2.0-rocm-5.4.3-py310-linux_x86_64.zip
unzip torch-2.0-rocm-5.4.3-py310-linux_x86_64.zip
cd torch-2.0-rocm-5.4.3-py310-linux_x86_64
pip install ./*
cd ..

# module avail
# pytorch-geometric/2.3.0-pytorch-1.13.0-rocm-5.3

# All
# after changing torch-geometric all datasets need to be reprocessed
rm -r -f datasets/*

# try
python -c "import torch; import torch; print('torch.__version__', torch.__version__); import torch_geometric; print('torch_geometric.__version__', torch_geometric.__version__)" 
python -c "print('torch.version.cuda', torch.version.cuda);"


# pip install torchdeq
cd torchdeq
pip install -e .
cd ..

pip install e3nn==0.4.4 timm==0.4.12

pip install matplotlib seaborn scikit-image
pip install hydra-core wandb omegaconf black

pip install numba sphinx nbsphinx sphinx-rtd-theme pandoc ase==3.21.* pre-commit==2.10.* tensorboard

pip install -e .

wandb login

# add an alias for easy run launching
echo 'alias launchrun="/u/andreasb/miniforge3/envs/deq/bin/python u/andreasb/equilibrium-forcefields/scripts/train_deq_md_v2.py"' >> ~/.bashrc
# or on a cluster
echo 'alias launchrun="sbatch /u/andreasb/equilibrium-forcefields/scripts/cslab_launcher.slrm train_deq_md_v2.py"' >> ~/.bashrc

# mamba env export > environment_slurm.yml
```

### Vector Slurm Cluster

Note: our slurm cluster has 
```ldd --version -> ldd (Ubuntu GLIBC 2.27-3ubuntu1.6) 2.27```
leading to
```OSError: /lib/x86_64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found (required by /venv10/lib/python3.10/site-packages/libpyg.so)```
So we need to install older version of python and pytorch

#### Create environment
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc

# export PYTHONPATH=$PYTHONPATH:/pkgs/python-3.9.10
mamba create -n deq python=3.9
# mamba create -n deq python='/pkgs/python-3.9.10'
mamba activate deq
```

```bash
sbatch scripts/amd_launcher.slrm train_deq_md_v2.py +use=deq +cfg=fpc_of +trial=test
```


#### Set environment variables (Vector cluster)

```bash
mamba env config vars list
mamba env config vars set CFLAGS="-I/usr/local/include -I/pkgs/cuda-12.1/targets/x86_64-linux/include $CFLAGS"
mamba env config vars set LDFLAGS="-L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -L/pkgs/cuda-12.1/targets/x86_64-linux/lib $LDFLAGS"
mamba env config vars set CUDA_HOME=/pkgs/cuda-12.1
mamba env config vars set CUDA_ROOT=/pkgs/cuda-12.1
mamba env config vars set LD_LIBRARY_PATH=/pkgs/cuda-12.1/lib64:/pkgs/cudnn-12.1-v8.2.4.15/lib64:$LD_LIBRARY_PATH
mamba env config vars set PATH=/usr/local/cuda-12.1/bin:$PATH
mamba env config vars set CPATH=$CPATH:/pkgs/cuda-12.1/include
mamba env config vars set CUDNN_PATH=/h/burgeran/miniforge3/envs/deq/lib/python3.9/site-packages/nvidia/cudnn
mamba env config vars set LD_LIBRARY_PATH=${CUDNN_PATH}/lib:$LD_LIBRARY_PATH
mamba activate deq
```
or
```bash
cd %CONDA_PREFIX%
mkdir ./etc/conda/activate.d
mkdir ./etc/conda/deactivate.d
```
```bash
# first check cuda version with $ nvcc --version or $ nvidia-smi
export CFLAGS="-I/usr/local/include -I/pkgs/cuda-12.1/targets/x86_64-linux/include $CFLAGS"
export LDFLAGS="-L/usr/local/lib -L/usr/lib/x86_64-linux-gnu -L/pkgs/cuda-12.1/targets/x86_64-linux/lib $LDFLAGS"

# cuda: find / -iname “cuda.h”
export CUDA_HOME=/pkgs/cuda-12.1
export CUDA_ROOT=/pkgs/cuda-12.1
export LD_LIBRARY_PATH=/pkgs/cuda-12.1/lib64:/pkgs/cudnn-12.1-v8.2.4.15/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.1/bin:$PATH
export CPATH=$CPATH:/pkgs/cuda-12.1/include
# add cudann to path
# export CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
# export CUDNN_PATH=/h/burgeran/skills/skillsenv10/lib/python3.10/site-packages/nvidia/cudnn
export CUDNN_PATH=/h/burgeran/miniforge3/envs/deq/lib/python3.9/site-packages/nvidia/cudnn
export LD_LIBRARY_PATH=${CUDNN_PATH}/lib:$LD_LIBRARY_PATH
```

#### glibc

Error
```bash
File "/h/burgeran/miniforge3/envs/deq/lib/python3.9/site-packages/torch_sparse/typing.py", line 2, in <module>
    import pyg_lib  # noqa
File "/h/burgeran/miniforge3/envs/deq/lib/python3.9/site-packages/pyg_lib/__init__.py", line 39, in <module>
load_library('libpyg')
File "/h/burgeran/miniforge3/envs/deq/lib/python3.9/site-packages/pyg_lib/__init__.py", line 36, in load_library
torch.ops.load_library(spec.origin)
File "/h/burgeran/miniforge3/envs/deq/lib/python3.9/site-packages/torch/_ops.py", line 933, in load_library
ctypes.CDLL(path)
File "/h/burgeran/miniforge3/envs/deq/lib/python3.9/ctypes/__init__.py", line 374, in __init__
    self._handle = _dlopen(self._name, mode)
OSError: /lib/x86_64-linux-gnu/libm.so.6: version GLIBC_2.29' not found (required by /fs01/home/burgeran/miniforge3/envs/deq/lib/python3.9/site-packages/libpyg.so)
```

```bash
# mamba install -c conda-forge glibc=2.29
mamba install rmg::glibc
```


```bash
wget http://ftp.gnu.org/gnu/libc/glibc-2.29.tar.gz
tar -xvzf glibc-2.29.tar.gz
cd glibc-2.29
mkdir build
cd build
unset LD_LIBRARY_PATH
# ../configure --prefix=$HOME/.local
CFLAGS="-O2" ../configure --prefix=$HOME/.local
make # -j4
make install
$HOME/glibc-2.29/lib/ld-2.29.so --version
```

```bash
/h/burgeran/glibc-2.29/build/elf/ldconfig: Warning: ignoring configuration file that cannot be opened: /h/burgeran/.local/etc/ld.so.conf: No such file or directory
make[1]: Leaving directory '/fs01/home/burgeran/glibc-2.29'
```


```bash
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
# add to bashrc
echo 'export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

#### export env
```bash
mamba activate deq
mamba env export | grep -v '^prefix:' | grep -v 'torch' | grep -v 'ocp-models' | grep -v 'equiformer' > environment.yml
```

## Compute canada

```bash
module load StdEnv/2023 
module load intel/2023.2.1
module load cuda/11.8
module load python/3.10
virtualenv --no-download deqenv
source deqenv/bin/activate
pip install --no-index --upgrade pip

# https://docs.alliancecan.ca/wiki/Python#Loading_a_Python_module
# avail_wheels "*cdf*"
# vail_wheels numpy==1.23
# avail_wheels 'numpy<1.23'
# avail_wheels 'numpy<1.23' --python 3.9
# avail_wheels -r requirements.txt 
# avail_wheels -r requirements.txt --not-available

pip install torch==2.2.0 torchvision==0.17.0 --no-index
pip install --no-index torch_scatter torch_sparse torch_cluster torch_spline_conv
pip install --no-index pyg_lib -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install --no-index torch-geometric==2.0.4 
# pip install torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

pip install e3nn==0.4.4 timm==0.4.12

pip install matplotlib seaborn scikit-image
pip install hydra-core wandb omegaconf black

pip install numba sphinx nbsphinx sphinx-rtd-theme pandoc ase==3.21.* pre-commit==2.10.* tensorboard
pip install setuptools==57.4.0
pip install demjson 
# pip install demjson3
pip install lmdb==1.1.1
pip install "ray[tune]"
pip install submitit


# Installing collected packages: opt-einsum, torch, opt-einsum-fx, timm, e3nn
#   Attempting uninstall: torch
#     Found existing installation: torch 2.3.1+computecanada
#     Uninstalling torch-2.3.1+computecanada:
#       Successfully uninstalled torch-2.3.1+computecanada
# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
# torch-cluster 1.6.3+torch23.computecanada requires torch~=2.3.0, but you have torch 2.2.0+computecanada which is incompatible.
# torch-scatter 2.1.2+torch23.computecanada requires torch~=2.3.0, but you have torch 2.2.0+computecanada which is incompatible.

```

## Training

train equiformer
```bash
cd equiformer

# QM9
sh scripts/train/qm9/equiformer/target@1.sh

# MD17
sh scripts/train/md17/equiformer/se_l2/target@aspirin.sh    # L_max = 2

# OC20 IS2RE 
sh scripts/train/oc20/is2re/graph_attention_transformer/l1_256_nonlinear_split@all_g@2.sh
```