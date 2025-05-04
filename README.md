# Objective

## How to train the reg-network
```
/home/USERNAME/code/DiffSplitting/time_prediction_training.py --config=config/splitting_hagen_time_predictor.json --rootdir=/group/training/diffsplit/ -enable_wandb
```

## How to train the indiSplit network.
```
/home/USERNAME/code/DiffSplitting/split.py -c /home/USERNAME/code/DiffSplitting/config/splitting_hagen_indi_joint.json -enable_wandb
```

## Evaluating a network
Run the `DiffSplitting/notebooks/EvaluateJointIndi.ipynb` notebook.