#!/bin/bash

# launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05
# launchrun +use=deq +cfg=fpc_of model.alpha_drop=0.1 model.drop_path_rate=0.05

# seed=2
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 seed=2 wandb_tags=["drop"]
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 seed=2 wandb_tags=["drop"]
launchrun +use=deq +cfg=fpc_of model.alpha_drop=0.1 model.drop_path_rate=0.05 seed=2 wandb_tags=["drop"]
# seed=3
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 seed=3 wandb_tags=["drop"]
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 seed=3 wandb_tags=["drop"]
launchrun +use=deq +cfg=fpc_of model.alpha_drop=0.1 model.drop_path_rate=0.05 seed=3 wandb_tags=["drop"]

# bigger molecule
# launchrun dname=md22 target=stachyose model.alpha_drop=0.1 model.drop_path_rate=0.05
# launchrun +use=deq +cfg=fpc_of dname=md22 target=stachyose model.drop_path_rate=0.05
# launchrun +use=deq +cfg=fpc_of dname=md22 target=stachyose model.alpha_drop=0.1 model.drop_path_rate=0.05
