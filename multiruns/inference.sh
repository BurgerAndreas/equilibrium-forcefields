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

# launchrun evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1
# launchrun evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4
# launchrun evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8

# md17=(benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)
# for target in "${md17[@]}"; do
#     # loop over fpreuse_f_tol
#     # 16 values * 8 molecules * 2 models * 3 seeds * 1 minute = 768 minutes = 12.8 hours
#     # for fpreuse_f_tol in 1e-4 1e-3 1e-2 5e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0 1e1 1e2; do
#     #     # without dropout
#     #     launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=$fpreuse_f_tol target=$target
#     #     launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 deq_kwargs_test.fpreuse_f_tol=$fpreuse_f_tol target=$target
#     #     # with dropout
#     #     # launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.drop_path_rate=0.05 model.num_layers=2 deq_kwargs_test.fpreuse_f_tol=$fpreuse_f_tol target=$target
#     # done

#     # DEQ most important run
#     launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=2e-1 target=$target
#     launchrun +use=deq +cfg=fpc_of model.num_layers=2 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=2e-1 target=$target

#     # Equiformer
#     launchrun evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=$target
#     launchrun evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=$target
#     launchrun evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=$target
# done

for target in benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil; do
    launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=$target
    
    launchrun +use=deq +cfg=fpc_of model.num_layers=2 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=2e-1 target=$target
    launchrun +use=deq +cfg=fpc_of model.num_layers=2 evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=1e-1 target=$target

    launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=2e-1 target=$target
    launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.fpreuse_f_tol=1e-1 target=$target

done
