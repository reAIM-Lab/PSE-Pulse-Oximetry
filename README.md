## Overview

This repo contains the implementation for "Path-specific effects for  pulse-oximetry guided decisions in critical care".

## Installation

Create a conda virtual environment and install the dependencies. All the experiments were performed on one NVIDIA RTX 6000 GPU, 32 CPU cores, 256GB of system RAM, running Ubuntu 22.04 with CUDA 12.2.

```
conda env create -f environment.yml
```

## Data Preparation

Navigate to the corresponding dataset directory in [data/eicu](data/eicu) or [data/mimic-iv](data/mimic-iv) and run the ```preprocess.sh``` and ```process.py``` scripts. For the eICU dataset, separate queries for the ventilation outcomes and Charlson Comorbidity Scores in Google BigQuery are included the notebook ```eicu_ventilation_charlson.ipynb```.

Validate all datasets by navigating to [experiments](experiments) and running ```datasets.py```.

## Training

Train all models by navigating to [experiments](experiments) and running ```main.py```. The results for each experiment will be stored in a separate csv file in [experiments/results](experiments/results).

## Plotting and analysis

Plot all experiment results by navigating to [experiments](experiments) and running ```plot.py```. For the dataset analysis, navigate to [data](data) and run ```viz.py```.