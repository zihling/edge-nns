# HGCal Autoencoder
This repo contains a small, medium, and large Autoencoder trained on the HGCal dataset.
These models were developed using a Bayesian optimization process, and the models were selected along a Pareto front defined by Model error versus Binary Operations (BOPS). 

Models provided:
* Small Pareto
* Medium Pareto
* Large Pareto

## Dataset
To evaluate the model, download the dataset [here](https://cseweb.ucsd.edu/~oweng/hgcal_dataset/keras_version/).

## Load the model
We provide a script to load the model for further evaluation: `./scripts/load_model.sh`. 
No need to download the dataset to load the model.

To load the Medium Pareto model, run:
```
./scripts/load_model.sh 0
```

To load the Small Pareto model, run:
```
./scripts/load_model.sh 1
```

To load the Large Pareto model, run:
```
./scripts/load_model.sh 2
```

## Evaluate a model
We provide a script for evaluating the Earth Mover's Distance (EMD) of the model: `./scripts/eval.sh`.
Lower EMD is better. 
A perfect autoencoder reconstruction of the data would yield an EMD of 0.

Make sure to point the `.scripts/eval.sh`'s `DATASET` path to where you have stored the dataset, e.g.,
```bash
DATASET=./data/pickled--data_values--phys_values--EoL_dataset.pkl
```

To evaluate the Medium Pareto model, run:
```bash
./scripts/eval.sh 0
```

To evaluate the Small Pareto model, run:
```bash
./scripts/eval.sh 1
```

To evaluate the Large Pareto model, run:
```bash
./scripts/eval.sh 2
```