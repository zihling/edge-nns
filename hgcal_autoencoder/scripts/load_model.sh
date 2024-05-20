#!/bin/bash

PATH_PREFIX=./model_ckpts
MODEL_ID=""
if [ $1 -eq 0 ]; then
	# ECONT-baseline / Medium Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/8x8_c8_S2_tele_fqK_6bit.hdf5
	MODEL_NAME=8x8_c8_S2_tele_fqK_6bit
	BIT_WIDTH=6
elif [ $1 -eq 1 ]; then
	# ECONT-small-pareto / Small Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/run3-row27-small-econ.hdf5
	MODEL_NAME=run3-row27-small-econ
	BIT_WIDTH=8
elif [ $1 -eq 2 ]; then
	# ECONT-large-pareto / Large Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/run5-row21-big-econ.hdf5
	MODEL_NAME=run5-row21-big-econ
else
	echo "Error"
fi

python3 load_model.py \
	--model $MODEL_NAME \
	--pretrained-model $PRETRAINED_MODEL