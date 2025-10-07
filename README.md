# Examples of edge neural networks
This repo contains examples of edge neural networks (NNs).

## Models 
We provide scripts for evaluating the following edge NNs:
* `HGCal Autoencoder`: Autoencoder used at the Large Hadron Collider's (LHC) Compact Muon Solenoid (CMS) experiment for compressing physics sensor data generated at the High Granularity Calorimeter (HGCal).
* `CIFAR-10 CNN`: Convolutional NN (CNN) that classifies images provided by the CIFAR-10 dataset. These models are based on the [MLPerf Tiny Benchmark](https://github.com/mlcommons/tiny/tree/master). 


## Create the environment
We use `conda` to manage our environment. 
Install `miniconda` from [here](https://docs.anaconda.com/free/miniconda/) if you do not already have `conda` installed.

Then create the environment:
```
conda env create -f environment.yml
```

Activate the environment:
```
conda activate egde-nns
```

For Python3.10+, try virtual environment:
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```