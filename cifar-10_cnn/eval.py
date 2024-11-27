import argparse
import tensorflow as tf
from load_model import load_model


NUM_CLASSES = 10

def main(args):
    model = load_model(args.config, pretrained_model=args.pretrained_model)
    print(model.summary())
    print()

    # Load the CIFAR10 dataset
    _, (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test / 256.
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    criterion = tf.keras.losses.categorical_crossentropy

    y_pred = model.predict(x_test)

    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_test shape: {y_test.shape}")

    loss = criterion(y_test, y_pred)
    acc = tf.keras.metrics.categorical_accuracy(y_test, y_pred)

    print(f"Loss: {loss.numpy().mean()}")
    print(f"Accuracy: {acc.numpy().mean()}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', 
        '--config', 
        type=str, 
        default="./model_ckpts/resnet_v1_eembc_quantized_baseline_fkeras/baseline_fkeras.yml", 
        help="specify yaml config"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="specify pretrained model file path",
    )
    args = parser.parse_args()
    main(args)