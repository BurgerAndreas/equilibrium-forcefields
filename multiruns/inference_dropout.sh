#!/bin/bash

# run this on a machine with the checkpointed models from multiruns/md17.sh
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 seed=2
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 seed=3
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 model.num_layers=2
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 model.num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 model.num_layers=2 seed=3
# launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 model.num_layers=1
# launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 model.num_layers=1 seed=2
# launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 model.num_layers=1 seed=3
# launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 model.num_layers=2 
# launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 model.num_layers=2 seed=2
# launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 model.num_layers=2 seed=3

# for speed table and accuracy table

# loop over molecules
# get_all_models
md17=(benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)

# launchrun evaluate=True model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1
# launchrun evaluate=True model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4
# launchrun evaluate=True model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8

# # dropout 
# for target in "${md17[@]}"; do

#     # DEQ most important run
#     launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=2e-1 target=$target
#     launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=1e-1 target=$target

#     launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 model.num_layers=2 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=1e-1 target=$target
#     launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 model.num_layers=2 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=2e-1 target=$target

#     # Equiformer
#     launchrun evaluate=True model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=$target
#     launchrun evaluate=True model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=$target
#     launchrun evaluate=True model.alpha_drop=0.1 model.drop_path_rate=0.05 wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=$target
# done

for target in "${md17[@]}"; do

    # DEQ most important run
    launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 target=$target 
    launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 model.num_layers=2 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1  target=$target

done