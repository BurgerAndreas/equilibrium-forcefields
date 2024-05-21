#!/bin/bash


# run this on a machine with the checkpointed models from multiruns/md17.sh
# launchrun +use=deq +cfg=fpc_of
# launchrun +use=deq +cfg=fpc_of seed=2
# launchrun +use=deq +cfg=fpc_of seed=3
# launchrun +use=deq +cfg=fpc_of model.num_layers=2
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 seed=2
# launchrun +use=deq +cfg=fpc_of model.num_layers=2 seed=3
# launchrun model.num_layers=1
# launchrun model.num_layers=1 seed=2
# launchrun model.num_layers=1 seed=3
# launchrun model.num_layers=2 
# launchrun model.num_layers=2 seed=2
# launchrun model.num_layers=2 seed=3

# ls models/md17/deq_equiformer_v2_oc20/aspirin/DEQE2fpcofdroppathrate005
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1

# seed=2
# ls models/md17/deq_equiformer_v2_oc20/aspirin/DEQE2fpcofdroppathrate005seed2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=2

# seed=3
# ls models/md17/deq_equiformer_v2_oc20/aspirin/DEQE2fpcofdroppathrate005seed3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 seed=3

# model.num_layers=2
# ls models/md17/deq_equiformer_v2_oc20/aspirin/DEQE2fpcofdroppathrate005numlayers2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2

# model.num_layers=2, seed=2
# ls models/md17/deq_equiformer_v2_oc20/aspirin/DEQE2fpcofdroppathrate005numlayers2seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=2
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=2

# model.num_layers=2, seed=3
# ls models/md17/deq_equiformer_v2_oc20/aspirin/DEQE2fpcofdroppathrate005numlayers3seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-5 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-4 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-3 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e-1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e1 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=3
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e2 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 seed=3


####################################################################################################
# Equiformer aspirin
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8

# seed=2
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 seed=2
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 seed=2
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 seed=2

# seed=3
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 seed=3
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 seed=3
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 seed=3


####################################################################################################
# Speed runs for all molecules at deq_kwargs_test.fpreuse_f_tol=1e-1
####################################################################################################

# md17=(aspirin benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 target=benzene
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 target=ethanol
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 target=malonaldehyde
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 target=naphthalene
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 target=salicylic_acid
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 target=toluene
launchrun +use=deq +cfg=fpc_of evaluate=True deq_kwargs_test.fpreuse_f_tol=1e0 wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=2 target=uracil


####################################################################################################
# Equiformer

launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=benzene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=ethanol
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=malonaldehyde
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=naphthalene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=salicylic_acid
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=toluene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=1 target=uracil

# model.num_layers=4
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=benzene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=ethanol
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=malonaldehyde
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=naphthalene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=salicylic_acid
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=toluene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=4 target=uracil


# model.num_layers=8
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=benzene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=ethanol
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=malonaldehyde
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=naphthalene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=salicylic_acid
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=toluene
launchrun evaluate=True wandb_tags=["inference"] assert_checkpoint=True eval_batch_size=1 model.num_layers=8 target=uracil


# md22=(AT_AT_CG_CG AT_AT Ac_Ala3_NHMe DHA buckyball_catcher dw_nanotube stachyose)

