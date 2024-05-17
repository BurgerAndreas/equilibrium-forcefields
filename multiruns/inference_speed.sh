#!/bin/bash

# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_speed"] assert_checkpoint=True

# launchrun evaluate=True model.num_layers=1 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun evaluate=True model.num_layers=2 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun evaluate=True model.num_layers=3 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun evaluate=True model.num_layers=4 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun evaluate=True model.num_layers=5 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun evaluate=True model.num_layers=6 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun evaluate=True model.num_layers=7 wandb_tags=["inference_speed"] assert_checkpoint=True
# launchrun evaluate=True model.num_layers=8 wandb_tags=["inference_speed"] assert_checkpoint=True

# same thing again with eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1

# launchrun evaluate=True model.num_layers=1 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun evaluate=True model.num_layers=2 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun evaluate=True model.num_layers=3 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun evaluate=True model.num_layers=4 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun evaluate=True model.num_layers=5 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun evaluate=True model.num_layers=6 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun evaluate=True model.num_layers=7 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1
# launchrun evaluate=True model.num_layers=8 wandb_tags=["inference_speed"] assert_checkpoint=True eval_batch_size=1

# seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2

# launchrun evaluate=True model.num_layers=1 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun evaluate=True model.num_layers=2 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun evaluate=True model.num_layers=3 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun evaluate=True model.num_layers=4 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun evaluate=True model.num_layers=5 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun evaluate=True model.num_layers=6 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun evaluate=True model.num_layers=7 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2
# launchrun evaluate=True model.num_layers=8 wandb_tags=["inference_speed"] assert_checkpoint=True seed=2

# seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-6 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3

# launchrun evaluate=True model.num_layers=1 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun evaluate=True model.num_layers=2 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun evaluate=True model.num_layers=3 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun evaluate=True model.num_layers=4 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun evaluate=True model.num_layers=5 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun evaluate=True model.num_layers=6 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun evaluate=True model.num_layers=7 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3
# launchrun evaluate=True model.num_layers=8 wandb_tags=["inference_speed"] assert_checkpoint=True seed=3