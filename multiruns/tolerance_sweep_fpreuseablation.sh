#!/bin/bash

# source multiruns/tolerance_sweep_fpreuseablation.sh

# detailed sweep for phase transition
for f_tol in 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2; do
    # launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.f_tol=$f_tol
    launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 deq_kwargs_test.f_tol=$f_tol fpreuse_test=False
    sleep 5s
    for seed in 2 3; do
        # launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 deq_kwargs_test.f_tol=$f_tol seed=$seed
        launchrun +use=deq +cfg=fpc_of evaluate=True wandb_tags=["inference2"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 deq_kwargs_test.f_tol=$f_tol seed=$seed fpreuse_test=False
        sleep 5s
    done
done