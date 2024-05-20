# HGCal Autoencoder
This repo contains a small, medium, and large Autoencoder trained on the HGCal dataset.
These models were developed using a Bayesian optimization process, and the models were selected along a Pareto front defined by Model error versus Binary Operations (BOPS). 

Models provided:
* Small Pareto
* Medium Pareto
* Large Pareto

## Dataset
To evaluate the model, download the dataset [here](https://cseweb.ucsd.edu/~oweng/hgcal_dataset/).

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