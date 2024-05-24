import os
import argparse

from denseCNN import denseCNN
from qDenseCNN import qDenseCNN

def build_model(model_name, pretrained_model=None):
    # import network architecture and loss function
    from networks import networks_by_name
    # select models to run
    model_info = None
    if model_name != "":
        for n in networks_by_name:
            if n["name"] == model_name:
                model_info = n
                break
    if model_info is None:
        raise ValueError("No model specified. Check --model for correctness.")
        
    nBits_encod = dict()
    nBits_encod  = {'total':  9, 'integer': 1,'keep_negative':0} # 0 to 2 range, 8 bit decimal 
        
    if not 'nBits_encod' in model_info['params'].keys():
        model_info['params'].update({'nBits_encod':nBits_encod})
            
    # re-use trained weights 
    model_info['ws'] = pretrained_model
    if model_info['ws'] != None:
        if os.path.exists(model_info['ws']):
            print(f"Found user input weights, using {model_info['ws']}")
        else:
            raise ValueError(f"Provided weights file doesn't exist. File not found error: {model_info['ws']}")
    return model_info

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

def load_model(model_name, pretrained_model=None):
    # Build model
    model_info = build_model(model_name, pretrained_model)
    model = model_setup(model_info)
    m_autoCNN, m_autoCNNen = model.get_models()
    if pretrained_model and model_info["ws"] == "":
        raise RuntimeError("No weights provided to preload into the model!")
    return m_autoCNNen


def main(args):
    model = load_model(args.model, pretrained_model=args.pretrained_model)
    print(model.summary())
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

    args = parser.parse_args()
    main(args)