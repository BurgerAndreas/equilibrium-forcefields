

## Installation
```bash
mamba env create -n deq python=3.10
mamba activate deq


# get the data from the Open Catalyst Project
git clone git@github.com:Open-Catalyst-Project/ocp.git
cd ocp
mamba env update --name deq --file env.common.yml --prune
pip install -e .
# Structure to Energy and Forces (S2EF) task
# 3.4GB (17GB uncompressed)
python scripts/download_data.py --task s2ef --split "2M" --num-workers 8 --ref-energy
cd ..

# install equiformer_v2
git clone git@github.com:atomicarchitects/equiformer_v2.git
cd equiformer_v2
pip install -e .
cd ..

# install equiformer
git clone git@github.com:atomicarchitects/equiformer.git
cd equiformer
pip install -e .
mamba env update --name deq --file env/env_equiformer.yml --prune
cd ..

# link dataset for equiformer
cd equiformer/datasets
mkdir oc20
cd oc20
ln -s ../../../ocp/data/is2re is2re
cd ../../..

# check your cuda version with nvidia-smi
# https://github.com/pyg-team/pytorch_geometric/issues/999#issuecomment-606565132
pip uninstall torch torchvision torch-cluster torch-geometric torch-scatter torch-sparse -y
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip install scikit-image 

pip install torchdeq

pip install hydra-core wandb omegaconf

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