import yaml
import argparse
import tensorflow as tf

from train import load_data, load_model

def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Build model
    model = load_model(config, pretrained_model=args.pretrained_model)

    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    y_pred = model.predict(X_test)

    loss = criterion(y_test, y_pred)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()(y_test, y_pred)

    print(f"Loss: {loss.numpy().mean()}")
    print(f"Accuracy: {acc.numpy().mean()}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./model_ckpts/resnet_v1_eembc_quantized_baseline_fkeras/baseline_fkeras.yml",
        help="specify yaml config",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="specify pretrained model file path",
    )
    args = parser.parse_args()

    main(args)
