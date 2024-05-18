#!/bin/bash

# run this on a machine with the checkpointed models from multiruns/md17.sh
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_acc"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_acc"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_acc"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_acc"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_acc"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_acc"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_acc"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_acc"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_acc"] assert_checkpoint=True

# # seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_acc"] assert_checkpoint=True seed=2

# # seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_acc"] assert_checkpoint=True seed=3

# # num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2

# # num_layers=2, seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=2

# # num_layers=2, seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_acc"] assert_checkpoint=True num_layers=2 seed=3
