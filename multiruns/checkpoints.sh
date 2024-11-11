#!/bin/bash


md17=(benzene ethanol malonaldehyde naphthalene salicylic_acid toluene uracil)

basename=DEQsaplnift

for target in "${md17[@]}"; do

    # check if the target directory exists
    # if target is aspirin
    if [ $target == "aspirin" ]; then
        ls models/md17/deq_equiformer_v2_oc20/$target/$basename
    else
        ls models/md17/deq_equiformer_v2_oc20/$target/$basename$target
    fi

    ls models/md17/deq_equiformer_v2_oc20/$target/Esddnumlayers8
    ls models/md17/deq_equiformer_v2_oc20/$target/Esddnumlayers8

done