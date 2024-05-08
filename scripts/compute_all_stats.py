import hydra
from omegaconf import DictConfig

import json
import pathlib
import pandas as pd
import yaml

import equiformer.datasets.pyg.md_all as md_all

import train_deq_md

# test:
# python scripts/train_deq_md.py compute_stats=True wandb=False
# python scripts/train_deq_md.py compute_stats=True wandb=False target=DHA dname=md22

# run this file:
# python scripts/compute_all_stats.py

def compute_all_stats(args: DictConfig) -> None:
    """Compute all statistics."""

    args.compute_stats = True

    for dname in ["md17", "md22"]:
        args.dname = dname
        molecules = list(md_all.MDAll.molecule_files[dname].keys())

        # try to load json if it exists
        fpath = pathlib.Path(__file__).parent.parent.absolute()
        fpath = fpath.joinpath("datasets", "statistics.json")
        if fpath.exists():
            statistics = json.load(open(fpath))
        else:
            statistics = {}
        
        if dname not in statistics:
            statistics[dname] = {}

        # compute statistics
        radii = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 12.0, 20.0]
        for max_radius in radii:
            args.model.max_radius = max_radius

            for mol in molecules:
                args.target = mol
                _stats = train_deq_md.main(args)
                if mol not in statistics[dname]:
                    statistics[dname][mol] = {}
                statistics[dname][mol][str(max_radius)] = _stats
        
        # print(f'statistics:\n{yaml.dump(statistics)}')
        
        # compute average over molecules
        # Flatten the nested dictionary and create a DataFrame 
        df = pd.DataFrame.from_dict({
            (level1, level2, level3): _stat 
            for level1, _mol in statistics.items() if _mol != "_avg"
            for level2, _radius in _mol.items() 
            for level3, _stat in _radius.items()
        }, orient='index') 

        # name the columns
        df.index = pd.MultiIndex.from_tuples(df.index, names=['dataset', 'molecule', 'max_radius'])        
        print(f'\ndf:\n{df}')

        radii = list(statistics[dname][molecules[0]].keys())
        stattypes = list(statistics[dname][molecules[0]][radii[0]].keys())

        if "_avg" not in statistics[dname]:
            statistics[dname]["_avg"] = {}
        for max_radius in radii:
            # average over all molecules
            _stat = df.loc[(dname, slice(None), max_radius), :].mean().to_dict()
            statistics[dname]["_avg"][max_radius] = _stat
        
        # save in datasets/statistics as json
        json.dump(statistics, open(fpath, "w"), indent=2)

        print(f"\nStatistics computed and saved to {fpath}.\n")
        print(yaml.dump(statistics))

@hydra.main(config_name="md17", config_path="../equiformer/config", version_base="1.3")
def hydra_wrapper(args: DictConfig) -> None:
    """Run training loop."""

    from deq2ff.logging_utils import init_wandb

    args.wandb = False
    init_wandb(args)

    compute_all_stats(args)

if __name__ == "__main__":
    # train_deq_md.hydra_wrapper()
    hydra_wrapper()