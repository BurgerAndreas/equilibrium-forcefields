#!/bin/bash

# TODO: try again with eval_batch_size=1?

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

launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True

# seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True seed=2

# seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True seed=3

# model.num_layers=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2

# model.num_layers=2, seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=2

# model.num_layers=2, seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of model.drop_path_rate=0.05 evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True model.num_layers=2 seed=3


####################################################################################################
# Equiformer
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=1
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=4
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=8

# seed=2
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=1 seed=2
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=4 seed=2
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=8 seed=2

# seed=3
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=1 seed=3
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=4 seed=3
launchrun model.alpha_drop=0.1 model.drop_path_rate=0.05 evaluate=True wandb_tags=["inference"] assert_checkpoint=True model.num_layers=8 seed=3


