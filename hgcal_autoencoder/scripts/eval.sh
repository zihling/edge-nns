#!/bin/bash

MODEL_TYPE=$1
DATASET=./data/eol_hgcal_dataset.pkl

if [ $MODEL_TYPE -eq 0 ]; then # Medium model
    MODEL=8x8_c8_S2_tele_fqK_6bit
    PRETRAINED_MODEL="./model_ckpts/8x8_c8_S2_tele_fqK_6bit.hdf5"
elif [ $MODEL_TYPE -eq 1 ]; then # Small model
    MODEL=run3-row27-small-econ
    PRETRAINED_MODEL="./model_ckpts/run3-row27-small-econ.hdf5"
elif [ $MODEL_TYPE -eq 2 ]; then # Large model
    MODEL=run5-row21-big-econ
    PRETRAINED_MODEL="./model_ckpts/run5-row21-big-econ.hdf5"
# QKeras models
elif [ $1 -eq 3 ]; then
	# ECONT-baseline / Medium Pareto
	PRETRAINED_MODEL=./model_ckpts/8x8_c8_S2_tele_qK_6bit.hdf5
	MODEL=8x8_c8_S2_tele_qK_6bit
elif [ $1 -eq 4 ]; then
	# ECONT-small-pareto / Small Pareto
	PRETRAINED_MODEL=./model_ckpts/run3-row27-small-econ-qkeras.hdf5
	MODEL=run3-row27-small-econ-qkeras
elif [ $1 -eq 5 ]; then
	# ECONT-large-pareto / Large Pareto
	PRETRAINED_MODEL=./model_ckpts/run5-row21-big-econ-qkeras.hdf5
	MODEL=run5-row21-big-econ-qkeras
else
    echo "Error"
fi

CUDA_VISIBLE_DEVICES="" python eval.py \
    --data $DATASET \
    --model $MODEL \
    --pretrained-model $PRETRAINED_MODEL