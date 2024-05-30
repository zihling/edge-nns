#!/bin/bash
PATH_PREFIX=./model_ckpts

if [ $1 -eq 0 ]; then
    # CIFAR-10 tiny2 CNN / Medium Model
	PRETRAINED_MODEL=$PATH_PREFIX/resnet_v1_eembc_quantized_tiny2_fkeras/resnet_v1_eembc_quantized_tiny2_fkeras.h5
	CONFIG=$PATH_PREFIX/resnet_v1_eembc_quantized_tiny2_fkeras/tiny2_pynq-z2_fkeras.yml
elif [ $1 -eq 1 ]; then
	# CIFAR-10 tiny CNN / Small Model
	PRETRAINED_MODEL=$PATH_PREFIX/resnet_v1_eembc_quantized_tiny_fkeras/resnet_v1_eembc_quantized_tiny_fkeras.h5
	CONFIG=$PATH_PREFIX/resnet_v1_eembc_quantized_tiny_fkeras/tiny_pynq-z2_fkeras.yml
elif [ $1 -eq 2 ]; then
	# CIFAR-10 baseline ResNet / Large model
	PRETRAINED_MODEL=$PATH_PREFIX/resnet_v1_eembc_quantized_baseline_fkeras/resnet_v1_eembc_quantized_baseline_fkeras.h5
	CONFIG=$PATH_PREFIX/resnet_v1_eembc_quantized_baseline_fkeras/baseline_fkeras.yml
else
	echo "Error"
fi

python load_model.py \
	--config $CONFIG \
	--pretrained-model $PRETRAINED_MODEL