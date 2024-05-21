#!/bin/bash

# source multiruns/inference_speed_accuracy_sweep.sh

# for fpreuse ablation

# # detailed sweep for phase transition
for fpreuse_f_tol in 1e-4 1e-3 1e-2 2e-2 3e-2 4e-2 5e-2 6e-2 7e-2 8e-2 9e-2 1e-1 2e-1 3e-1 4e-1 5e-1 6e-1 7e-1 8e-1 9e-1 1e0 1e1 1e2; do
    launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 deq_kwargs_test.fpreuse_f_tol=$fpreuse_f_tol
    sleep 5s
    for seed in 2 3; do
        launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 deq_kwargs_test.fpreuse_f_tol=$fpreuse_f_tol seed=$seed
        sleep 5s
    done
done
