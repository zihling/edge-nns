#!/bin/bash

PATH_PREFIX=./model_ckpts
MODEL_ID=""

# QKeras versions
if [ $1 -eq 0 ]; then
	# ECONT-baseline / Medium Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/8x8_c8_S2_tele_qK_6bit.hdf5
	MODEL_NAME=8x8_c8_S2_tele_qK_6bit
elif [ $1 -eq 1 ]; then
	# ECONT-small-pareto / Small Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/run3-row27-small-econ-qkeras.hdf5
	MODEL_NAME=run3-row27-small-econ-qkeras
elif [ $1 -eq 2 ]; then
	# ECONT-large-pareto / Large Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/run5-row21-big-econ-qkeras.hdf5
	MODEL_NAME=run5-row21-big-econ-qkeras
# FKeras versions
elif [ $1 -eq 3 ]; then
	# ECONT-baseline / Medium Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/8x8_c8_S2_tele_fqK_6bit.hdf5
	MODEL_NAME=8x8_c8_S2_tele_fqK_6bit
elif [ $1 -eq 4 ]; then
	# ECONT-small-pareto / Small Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/run3-row27-small-econ.hdf5
	MODEL_NAME=run3-row27-small-econ
elif [ $1 -eq 5 ]; then
	# ECONT-large-pareto / Large Pareto
	PRETRAINED_MODEL=$PATH_PREFIX/run5-row21-big-econ.hdf5
	MODEL_NAME=run5-row21-big-econ
else
	echo "Error"
fi

python load_model.py \
	--model $MODEL_NAME \
	--pretrained-model $PRETRAINED_MODEL