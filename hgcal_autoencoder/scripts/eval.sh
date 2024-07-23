#!/bin/bash

MODEL_TYPE=$1
DATASET=./data/pickled--data_values--phys_values--EoL_dataset.pkl

if [ $MODEL_TYPE -eq 0 ]; then # Medium model
    MODEL=8x8_c8_S2_tele_fqK_6bit
    PRETRAINED_MODEL="./model_ckpts/8x8_c8_S2_tele_fqK_6bit.hdf5"
elif [ $MODEL_TYPE -eq 1 ]; then # Small model
    MODEL=run3-row27-small-econ
    PRETRAINED_MODEL="./model_ckpts/run3-row27-small-econ.hdf5"
elif [ $MODEL_TYPE -eq 2 ]; then # Large model
    MODEL=run5-row21-big-econ
    PRETRAINED_MODEL="./model_ckpts/run5-row21-big-econ.hdf5"
else
    echo "Error"
fi

python eval.py \
    --data $DATASET \
    --model $MODEL \
    --pretrained-model $PRETRAINED_MODEL