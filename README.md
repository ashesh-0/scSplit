# scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy

## Overview

This is the official implementation of **scSplit: Bringing Severity Cognizance to Image Decomposition in Fluorescence Microscopy**, published at NeurIPS 2025. [Paper link](https://arxiv.org/abs/2503.22983).

## Installation
1. Install [mamba](https://github.com/mamba-org/mamba).
2. Execute installation.sh.

After installation, one needs to do `mamba activate split_hpc` to activate the environment and one can subsequently start to train or evaluate the models. 

## Training
### For $Gen_i$ networks
```
python /home/ashesh.ashesh/code/DiffSplitting/split.py -c /home/ashesh.ashesh/code/DiffSplitting/config/hagen_indiSplit.json -enable_wandb
```

### For $Reg$ network
```
python /home/ashesh.ashesh/code/DiffSplitting/time_prediction_training.py  --config=/home/ashesh.ashesh/code/DiffSplitting/config/ht_t24_time_predictor.json -enable_wandb
```

## Evaluation
Evaluation is done by running the notebooks. For some of the tasks, it can take more time and so we execute the notebooks as scripts.

### Synthetically Summed Inputs

In this case the argument `--mixing_t_ood=0.5` fixes the desired mixing-ratio in the inputs.
```
python notebooks/evaluate_notebook.py  --mmse_count=10  --notebook=/home/ashesh.ashesh/code/DiffSplitting/notebooks/EvaluateJointIndi.ipynb --ckpt=2502/BioSR-joint_indi-l1/5 --mixing_t_ood=0.5  --ckpt_time_predictor=2502/BioSR-UnetClassifier-l2/4 --outputdir=/group/jug/ashesh/indiSplit/notebook_results/
```

### Directly Imaged Inputs
```
python notebooks/evaluate_notebook.py --enable_real_input=true --ckpt=2502/HT_LIF24-joint_indi-l1/60 --mmse_count=10  --notebook=/home/ashesh.ashesh/code/DiffSplitting/notebooks/EvaluateJointIndiRealInput.ipynb --ckpt_time_predictor=2502/HT_LIF24-UnetClassifier-l2/3 
```


## Datasets
The publicly available datasets were utilized in our study. We adopted the same train/validation/test splits as those employed by previous methods focused on semantic unmixing. For convenience and clarity, we also provide these train/validation/test splits here.
- [Hagen et al.](TODO)
- [BioSR](TODO)
- [HTT24](TODO)
- [HTLIF24](TODO)
- [PaviaATN](TODO)

## Pre-trained models
- [Hagen et al.](TODO)
- [BioSR](TODO)
- [HTT24](TODO)
- [HTLIF24](TODO)
- [PaviaATN](TODO)