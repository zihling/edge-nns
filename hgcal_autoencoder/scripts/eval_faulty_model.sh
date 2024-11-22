#!/bin/bash

MODEL_TYPE=$1
BIT=$2
DATASET=./data/eol_hgcal_dataset.pkl

if [ $MODEL_TYPE -eq 0 ]; then # Medium model
    MODEL=8x8_c8_S2_tele_fqK_6bit
    PRETRAINED_MODEL="../../econ-t/fastml-science/sensor-data-compression/faulty_models/faulty_ECONT-baseline_bit${BIT}.hdf5"
elif [ $MODEL_TYPE -eq 1 ]; then # Small model
    MODEL=run3-row27-small-econ
    PRETRAINED_MODEL="../../econ-t/fastml-science/sensor-data-compression/faulty_models/faulty_ECONT-small-pareto_bit${BIT}.hdf5"
elif [ $MODEL_TYPE -eq 2 ]; then # Large model
    MODEL=run5-row21-big-econ
    PRETRAINED_MODEL="../../econ-t/fastml-science/sensor-data-compression/faulty_models/faulty_ECONT-large-pareto_bit${BIT}.hdf5"
else
    echo "Error"
fi

python eval.py \
    --data $DATASET \
    --model $MODEL \
    --pretrained-model $PRETRAINED_MODEL \