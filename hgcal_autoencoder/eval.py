import os
import time
import codecs
import pickle
import argparse
import numpy as np
import tensorflow as tf
from multiprocessing import Pool

from metrics import emd_multiproc
from load_model import load_model

from telescope import telescopeMSE8x8

def normalize(data, rescaleInputToMax=False, sumlog2=True):
    maxes = []
    sums = []
    sums_log2 = []
    for i in range(len(data)):
        maxes.append( data[i].max() )
        sums.append( data[i].sum() )
        sums_log2.append( 2**(np.floor(np.log2(data[i].sum()))) )
        if sumlog2:
            data[i] = 1.*data[i]/(sums_log2[-1] if sums_log2[-1] else 1.)
        elif rescaleInputToMax:
            data[i] = 1.*data[i]/(data[i].max() if data[i].max() else 1.)
        else:
            data[i] = 1.*data[i]/(data[i].sum() if data[i].sum() else 1.)
    if sumlog2:
        return  data,np.array(maxes),np.array(sums_log2)
    else:
        return data,np.array(maxes),np.array(sums)

def normalize_data(data_values):
    # normalize input charge data rescaleInputToMax: normalizes charges to
    # maximum charge in module sumlog2 (default): normalizes charges to
    # 2**floor(log2(sum of charge in module)) where floor is the largest scalar
    # integer: i.e. normalizes to MSB of the sum of charges (MSB here is the
    # most significant bit) rescaleSum: normalizes charges to sum of charge in
    # module
    norm_data, max_data, sum_data = normalize(
        data_values.copy(), rescaleInputToMax=False, sumlog2=True
    )

    return norm_data, max_data, sum_data

def unnormalize(norm_data,maxvals,rescaleOutputToMax=False, sumlog2=True):
    for i in range(len(norm_data)):
        if rescaleOutputToMax:
            norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].max() if norm_data[i].max() else 1.)
        else:
            if sumlog2:
                sumlog2 = 2**(np.floor(np.log2(norm_data[i].sum())))
                norm_data[i] =  norm_data[i] * maxvals[i] / (sumlog2 if sumlog2 else 1.)
            else:
                norm_data[i] =  norm_data[i] * maxvals[i] / (norm_data[i].sum() if norm_data[i].sum() else 1.)
    return norm_data

def split(shaped_data, validation_frac=0.2, randomize=False):
    N = round(len(shaped_data)*validation_frac)
    if randomize:
        val_index = np.random.choice(shaped_data.shape[0], N, replace=False) # randomly select 25% entries
        full_index = np.array(range(0,len(shaped_data))) # select the indices of the other 75%
        train_index = np.logical_not(np.in1d(full_index,val_index))

        val_input = shaped_data[val_index]
        train_input = shaped_data[train_index]
    else:
        val_input = shaped_data[:N]
        train_input = shaped_data[N:]
        val_index = np.arange(N)
        train_index = np.arange(len(shaped_data))[N:]

    print('Training shape')
    print(train_input.shape)
    print('Validation shape')
    print(val_input.shape)
    return val_input,train_input,val_index,train_index

def evaluate_model(model):
    """
    Evaluate model by calculating its loss and EMD
    """
    pass

def main(args):
    tf.keras.backend.clear_session()
    tf.config.run_functions_eagerly(True)
    # Load keras hdf5 model
    _, qdnn_model = load_model(args.model, pretrained_model=args.pretrained_model)

    # print(qdnn_model.summary())
    # Load and prep data
    ld_data_time_s = time.time()
    pickled_obj = "" 
    with open(args.data, "r") as f:
        pickled_obj = f.read()
    data_values, _ = pickle.loads(codecs.decode(pickled_obj.encode(), "base64"))
    ld_data_time = time.time() - ld_data_time_s
    print(f"Time to load data: {ld_data_time} seconds")

    normdata, maxdata, sumdata = normalize_data(data_values)
    maxdata = maxdata / 35.0  # normalize to units of transverse MIPs
    sumdata = sumdata / 35.0  # normalize to units of transverse MIPs

    # split in training/validation datasets
    shaped_data = qdnn_model.prepInput(normdata)
    val_input, _, val_ind, _ = split(shaped_data)
    val_sum = sumdata[val_ind]

    # Evaluate model by caluclating its loss and EMD
    print("Evaluating model...")
    curr_val_input = val_input[:20_000]
    autoencoder = qdnn_model.get_models()[0]
    # Predict
    cnn_deQ = autoencoder.predict(curr_val_input, batch_size=512)
    # Compute loss
    print("Computing loss...")
    print(f"cnn_deQ shape: {cnn_deQ.shape}")
    loss = telescopeMSE8x8(curr_val_input, cnn_deQ) # CORRECT
    print(f"Loss: {loss.numpy().mean()}")
    # Prep data for EMD calculation
    input_Q = curr_val_input
    input_calQ = qdnn_model.mapToCalQ(input_Q)  # shape = (N,48) in CALQ order
    output_calQ_fr = qdnn_model.mapToCalQ(cnn_deQ)  # shape = (N,48) in CALQ order
    print("Restore normalization")
    input_calQ = np.array(
        [
            input_calQ[i] * val_sum[i]
            for i in range(0, len(input_calQ))
        ]
    )  # shape = (N,48) in CALQ order
    output_calQ = unnormalize(
        output_calQ_fr.copy(),
        val_sum,
        rescaleOutputToMax=False,
    )
    with Pool() as pool:
        vals = pool.starmap(emd_multiproc, zip(input_calQ, output_calQ))
    mean_emd = np.mean(np.array(vals))
    print(f"Mean EMD: {mean_emd}")

    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="8x8_c8_S2_tele",
        dest="model",
        help="Model to run",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="path to pretrained model .hdf5 file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="pickled--data_values--phys_values--EoL_dataset.pkl",
        help="path to pickled data file",
    )

    args = parser.parse_args()
    main(args)