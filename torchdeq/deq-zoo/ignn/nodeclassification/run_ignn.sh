python train_IGNN_PPI.py      \
    --f_solver fixed_point_iter   \
    --ln_type spectral_norm \
    --norm_clip     \
    --grad 5        \
    --tau 0.5       \
    ${@:1}

