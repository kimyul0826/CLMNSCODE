# Optuna Hyperparameter Tuning Guide

## Overview

This guide explains how to use Optuna for hyperparameter tuning in the classification model training pipeline.

## Basic Usage

### 1. Default Tuning (No Config File)

```bash
python main.py --config config.yaml --mode tune --trials 20
```

This uses default hyperparameter ranges:
- Learning rate: 1e-5 to 1e-2 (loguniform)
- Batch size: [8, 16, 32, 64] (categorical)
- Epochs: [50, 100, 150, 200] (categorical)

### 2. Custom Tuning with Config File

```bash
python main.py --config config.yaml --mode tune --tune-config tune_config.yaml --trials 50
```

## Tuning Configuration File Format

Create a YAML file (e.g., `tune_config.yaml`) with the following structure:

```yaml
hyperparameters:
  # Learning rate search space
  learning_rate:
    type: loguniform  # loguniform, uniform
    low: 1e-5         # minimum value
    high: 1e-2        # maximum value
  
  # Batch size search space
  batch_size:
    type: categorical  # categorical, int
    choices: [8, 16, 32, 64, 128]  # possible values
  
  # Epochs search space
  epochs:
    type: categorical  # categorical, int
    choices: [50, 100, 150, 200, 300]  # possible values
  
  # Weight decay search space (optional)
  weight_decay:
    type: loguniform  # loguniform, uniform
    low: 1e-6         # minimum value
    high: 1e-3        # maximum value
  
  # Dropout rate search space (optional)
  dropout:
    type: uniform     # uniform
    low: 0.1          # minimum value
    high: 0.5         # maximum value

# Tuning settings (optional, can be overridden by command line)
tuning:
  trials: 50          # number of trials
  direction: maximize # maximize, minimize
  pruning: true       # enable pruning
  final_epochs: 300   # epochs for final training with best params
```

## Supported Hyperparameter Types

### 1. Continuous Parameters

#### Loguniform Distribution
```yaml
learning_rate:
  type: loguniform
  low: 1e-5
  high: 1e-2
```

#### Uniform Distribution
```yaml
dropout:
  type: uniform
  low: 0.1
  high: 0.5
```

### 2. Categorical Parameters

```yaml
batch_size:
  type: categorical
  choices: [8, 16, 32, 64, 128]
```

## Advanced Examples

### Example 1: Wide Search Space

```yaml
hyperparameters:
  learning_rate:
    type: loguniform
    low: 1e-6
    high: 1e-1
  
  batch_size:
    type: categorical
    choices: [4, 8, 16, 32, 64, 128]
  
  epochs:
    type: categorical
    choices: [25, 50, 100, 200, 400]
  
  weight_decay:
    type: loguniform
    low: 1e-7
    high: 1e-2
```

### Example 2: Focused Search Space

```yaml
hyperparameters:
  learning_rate:
    type: loguniform
    low: 1e-4
    high: 1e-2
  
  batch_size:
    type: categorical
    choices: [16, 32, 64]
  
  epochs:
    type: categorical
    choices: [100, 150, 200]
```

### Example 3: Model-Specific Parameters

```yaml
hyperparameters:
  learning_rate:
    type: loguniform
    low: 1e-5
    high: 1e-2
  
  batch_size:
    type: categorical
    choices: [8, 16, 32, 64]
  
  dropout:
    type: uniform
    low: 0.1
    high: 0.5
  
  weight_decay:
    type: loguniform
    low: 1e-6
    high: 1e-3
```

## Command Line Options

### Basic Tuning Options

```bash
--trials 50                    # Number of trials
--tune_pruning                 # Enable pruning
--tune_direction maximize      # Optimization direction
--tune_epochs 100              # Override epochs for tuning
--tune_final_epochs 300        # Epochs for final training
--tune_es_patience 20          # Early stopping patience
--tune_es_min_delta 0.0        # Early stopping min delta
```

### Complete Example

```bash
python main.py \
  --config config.yaml \
  --mode tune \
  --tune-config tune_config.yaml \
  --trials 100 \
  --tune_pruning \
  --tune_direction maximize \
  --tune_epochs 150 \
  --tune_final_epochs 500
```

## Output Structure

After tuning, you'll find:

```
runs/tune/experiment_name_YYYYMMDD_HHMMSS/
├── trial_000/                 # Individual trial results
├── trial_001/
├── ...
├── trial_099/
├── tuning_results.txt         # Best parameters summary
├── tuning_results.json        # Detailed results
└── final_training/            # Final training with best params
    ├── models/
    ├── plots/
    └── results/
```

## Best Practices

1. **Start with Default Ranges**: Use default ranges first to understand the problem
2. **Use Loguniform for Learning Rate**: Learning rates often work better with loguniform distribution
3. **Enable Pruning**: Use `--tune_pruning` to stop unpromising trials early
4. **Monitor Resources**: Large search spaces can be computationally expensive
5. **Save Results**: Always save tuning results for future reference

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size choices or use smaller models
2. **Slow Convergence**: Increase trials or adjust learning rate range
3. **Poor Results**: Check if the search space is appropriate for your dataset

### Tips

- Start with fewer trials (10-20) to test your configuration
- Use pruning to save time on unpromising trials
- Monitor GPU memory usage during tuning
- Save the best configuration for future use 