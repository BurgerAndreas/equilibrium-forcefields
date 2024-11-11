# equilibrium-forcefields
Scaling Force Fields with Equivariant Deep Equilibrium Models

Perks of Equilibrium Models:
- fast inference thanks to fixed-point reuse
- similar accuracy on MD17/MD22, better accuracy OC20
- less memeory use via memory-efficient backpropagation using implicit differentiation
- much fewer parameters (2-4x)

## Quickstart

Baseline Equiformer V2
```bash
python scripts/train_deq_oc20_v2.py model.num_layers=4
sbatch scripts/slurm_launcher.slrm train_deq_oc20_v2.py model.num_layers=4
```

DEQ Equiformer V2
```bash
python scripts/train_deq_oc20_v2.py +use=deq 
sbatch scripts/slurm_launcher.slrm train_deq_oc20_v2.py +use=deq
```

On MD17
```bash
python scripts/train_deq_md_v2.py target=aspirin
sbatch scripts/slurm_launcher.slrm train_deq_md_v2.py target=aspirin
```
