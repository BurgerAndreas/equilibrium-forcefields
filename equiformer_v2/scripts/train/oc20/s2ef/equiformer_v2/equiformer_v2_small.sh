python main_oc20.py \
    --distributed \
    --num-gpus 1 \
    --num-nodes 1 \
    --mode train \
    --config-yml 'oc20/configs/s2ef/all_md/equiformer_v2_small.yml' \
    --run-dir 'models/oc20/s2ef/all_md/equiformer_v2/N@20_L@6_M@3_153M/bs@512_lr@4e-4_wd@1e-3_epochs@1_warmup-epochs@0.01_g@8x16' \
    --print-every 200 \
    --amp \
    --submit
