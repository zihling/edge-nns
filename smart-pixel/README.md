# Smart Pixel Classification
This repo contains a small, medium, and 2 large dense NN trained on the Smart Pixel dataset.

Models provided:
* Small 
* Medium
* Large
* Large2

## Dataset
To evaluate the model, download the dataset [here](https://cseweb.ucsd.edu/~oweng/smart_pixel_dataset/).

Decompress the tar in a directory named `./data`.

## Train the model
We provide a script to train the model: `./scripts/train.sh`. 

### FKeras versions
To train the Medium model, run:
```
./scripts/train.sh 0
```

To train the Small model, run:
```
./scripts/train.sh 1
```

To train the Large model, run:
```
./scripts/train.sh 2
```

To train the Large2 model, run:
```
./scripts/train.sh 3
```

## Evaluate a model
We provide a script for evaluating the accuracy of the model: `./scripts/eval.sh`.

### FKeras versions
To evaluate the Medium model, run:
```bash
./scripts/eval.sh 0
```

To evaluate the Small model, run:
```bash
./scripts/eval.sh 1
```

To evaluate the Large2 model, run:
```bash
./scripts/eval.sh 2
```
