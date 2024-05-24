
# Pareto front: accuracy vs speed 
# different points: dew_kwargs_test.f_tol

############################################################################################################################################
# No FP reuse
############################################################################################################################################
# seed=1
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-5 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-4 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=3 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=4 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-1 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=5 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e0 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=6 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e1 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=7 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e2 evaluate=True

# seed=2
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-5 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-4 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-3 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=3 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-2 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=4 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-1 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=5 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e0 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=6 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e1 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=7 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e2 seed=2 evaluate=True

# seed=3
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-5 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-4 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-3 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=3 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-2 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=4 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e-1 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=5 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e0 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=6 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e1 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=7 model.path_drop=0.05 wandb_tags=["fpr_abl"] fpreuse_test=False dew_kwargs_test.f_tol=1e2 seed=3 evaluate=True


############################################################################################################################################
# FP reuse but do not compute first FP at full precision
############################################################################################################################################
# seed=1
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-5 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-4 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=3 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=4 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-1 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=5 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e0 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=6 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e1 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=7 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e2 evaluate=True

# seed=2
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-5 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-4 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-3 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=3 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-2 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=4 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-1 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=5 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e0 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=6 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e1 seed=2 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=7 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e2 seed=2 evaluate=True

# seed=3
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-5 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-4 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=2 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-3 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=3 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-2 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=4 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e-1 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=5 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e0 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=6 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e1 seed=3 evaluate=True
launchrun +use=deq +cfg=fpc_of model.num_layers=7 model.path_drop=0.05 wandb_tags=["fpr_abl"] dew_kwargs_test.f_tol=1e2 seed=3 evaluate=True