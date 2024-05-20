import os
import argparse
from pathlib import Path

from denseCNN import denseCNN
from qDenseCNN import qDenseCNN

def build_model(args):
    # import network architecture and loss function
    from networks import networks_by_name
    # select models to run
    model = None
    if args.model != "":
        for n in networks_by_name:
            if n["name"] == args.model:
                model = n
    if model is None:
        raise ValueError("No model specified. Check --model for correctness.")
        
    nBits_encod = dict()
    nBits_encod  = {'total':  9, 'integer': 1,'keep_negative':0} # 0 to 2 range, 8 bit decimal 
        
    if not 'nBits_encod' in model['params'].keys():
        model['params'].update({'nBits_encod':nBits_encod})
            
    # re-use trained weights 
    model['ws'] = args.pretrained_model
    if model['ws'] != "":
        if os.path.exists(model['ws']):
            print(f"Found user input weights, using {model['ws']}")
        else:
            raise ValueError(f"Provided weights file doesn't exist. File not found error: {model['ws']}")
    return model

def model_setup(model_info):
    if model_info["isQK"]:
        print("Model is a qDenseCNN")
        m = qDenseCNN(weights_f=model_info["ws"])
    else:
        print.info("Model is a denseCNN")
        m = denseCNN(weights_f=model_info["ws"])
    m.setpams(model_info["params"])
    m.init()
    return m


def main(args):
    # Build model
    model_info = build_model(args)
    model = model_setup(model_info)
    m_autoCNN, m_autoCNNen = model.get_models()
    model_info["m_autoCNN"] = m_autoCNN
    model_info["m_autoCNNen"] = m_autoCNNen # encoder only
    if model_info["ws"] == "":
        raise RuntimeError("No weights provided to preload into the model!")

    print(m_autoCNNen.summary())
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
        default="",
        help="path to pretrained model .hdf5 file",
    )

    args = parser.parse_args()
    main(args)