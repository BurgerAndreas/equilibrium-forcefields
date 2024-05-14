
replace `launchrun` with 
`sbatch scripts/amd_launcher.slrm train_deq_md_v2.py` 
or 
`sbatch scripts/slurm_launcher.slrm train_deq_md_v2.py`
or
`python scripts/train_deq_md_v2.py`

The easiest way is to define a function 
```bash
nano ~/.bashrc
```
and add the following line
```bash
launchrun() { sbatch scripts/amd_launcher.slrm "$@"; }
```
then run 
```bash
source ~/.bashrc
mamba activate deq
```
