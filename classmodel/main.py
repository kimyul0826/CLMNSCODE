#!/usr/bin/env python3
"""
Main script for model selection and execution control
"""

import argparse
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.config import (load_config, get_dataset_info, get_model_info, 
                         get_training_info, get_output_info, get_experiment_output_dir,
                         get_test_experiment_output_dir, find_latest_train_experiment_dir)
from models.resnet import ResNet18, ResNet50
from models.efficientnet import EfficientNet
from models.mobilenet import MobileNet
from train import train_model
from evaluate import evaluate_model
from utils.plot import plot_training_history, plot_confusion_matrix


def get_model(model_name, num_classes, pretrained=True):
    """
    Get model based on name
    
    Args:
        model_name: Name of the model
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        model: PyTorch model
    """
    if model_name == 'resnet18':
        return ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet50':
        return ResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet':
        return EfficientNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenet':
        return MobileNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description='Model Training, Evaluation, and Tuning')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file (yaml)')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'train_evaluate', 'tune'],
                       help='Mode to run')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model for evaluation')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name (overrides config)')
    parser.add_argument('--override', action='store_true',
                       help='Override config settings with command line arguments')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    # Tuning specific
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of Optuna trials when mode is tune')
    parser.add_argument('--tune_pruning', action='store_true',
                       help='Enable Optuna pruning during tuning')
    parser.add_argument('--tune_epochs', type=int, default=None,
                       help='Override epochs only for tuning (e.g., shorter runs)')
    parser.add_argument('--tune_direction', type=str, default='maximize', choices=['maximize', 'minimize'],
                       help='Optimization direction for Optuna (validation accuracy is maximized by default)')
    parser.add_argument('--tune_final_epochs', type=int, default=300,
                       help='Final train_evaluate epochs with best params')
    parser.add_argument('--tune_es_patience', type=int, default=20,
                       help='Early stopping patience used in final train_evaluate')
    parser.add_argument('--tune_es_min_delta', type=float, default=0.0,
                       help='Minimum improvement to reset patience in early stopping')

    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override experiment name if provided
    if args.name:
        config['output']['experiment_name'] = args.name
    
    # Get configuration sections
    dataset_info = get_dataset_info(config)
    model_info = get_model_info(config)
    training_info = get_training_info(config)
    output_info = get_output_info(config)
    
    # Override config settings if requested
    if args.override:
        if args.epochs is not None:
            training_info['epochs'] = args.epochs
        if args.batch_size is not None:
            training_info['batch_size'] = args.batch_size
        if args.lr is not None:
            training_info['learning_rate'] = args.lr
    
    
    
    # Get experiment directory based on mode
    if args.mode in ['train', 'train_evaluate']:
        experiment_dir = get_experiment_output_dir(config)
    elif args.mode in ['evaluate']:
        # In evaluate mode, do NOT create the directory yet to avoid duplicate "_1" suffixes.
        # Instead, derive the intended path and let evaluate_model create it.
        base_name = config['output']['experiment_name']
        fixed_subdir = None
        if args.model_path:
            mp = Path(args.model_path)
            # Derive base and timestamped subdir from provided model_path string (even if it doesn't exist)
            # Expect: runs/train/{base}/{base_YYMMDD_HHMMSS}/models/... OR runs/train/{base_YYMMDD_HHMMSS}/models/...
            timestamped = mp.parent.parent.name
            # Parent 3 may be the base name or 'train'
            parent3 = mp.parent.parent.parent.name
            # Helper to derive base from timestamped folder name
            def _derive_base_from_timestamped(name: str) -> str:
                parts = name.split('_')
                if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit() and len(parts[-1]) == 6 and len(parts[-2]) == 6:
                    return '_'.join(parts[:-2]) if len(parts[:-2]) > 0 else name
                return name
            if parent3 in {"train", "runs", ""}:
                base_name = _derive_base_from_timestamped(timestamped)
            else:
                base_name = parent3
            fixed_subdir = timestamped
            config['output']['experiment_name'] = base_name
            config['_fixed_test_subdir_name'] = fixed_subdir
        else:
            # No explicit model_path: follow the latest matching training run
            latest_dir = find_latest_train_experiment_dir(base_name)
            if latest_dir is not None:
                fixed_subdir = latest_dir.name
                config['_fixed_test_subdir_name'] = fixed_subdir

        # Preview the directory path without creating it
        experiment_dir = Path('runs') / 'test' / base_name / (fixed_subdir if fixed_subdir else f"{base_name}")
    else:
        # tune mode: create a parent directory for tuning runs
        base_name = config['output']['experiment_name']
        tune_root = Path('runs') / 'tune' / base_name
        tune_root.mkdir(parents=True, exist_ok=True)
        # timestamped tuning session directory
        from utils.config import _get_kst_timestamp
        tune_session_dir = tune_root / f"{base_name}_{_get_kst_timestamp()}"
        tune_session_dir.mkdir(parents=True, exist_ok=True)
        experiment_dir = tune_session_dir
    
    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"Model: {model_info['name']}")
    print(f"Classes: {dataset_info['classes']}")
    print(f"Number of classes: {dataset_info['num_classes']}")
    print(f"Training epochs: {training_info['epochs']}")
    print(f"Batch size: {training_info['batch_size']}")
    print(f"Learning rate: {training_info['learning_rate']}")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Transform type: {dataset_info['augmentation'].get('transform_type', 'standard')}")
    
    # Get model (for non-tune modes)
    if args.mode != 'tune':
        model = get_model(model_info['name'], dataset_info['num_classes'], model_info['pretrained'])
    
    if args.mode in ['train', 'train_evaluate']:
        print(f"\n🚀 Training {model_info['name']} model...")
        history = train_model(
            model=model,
            config=config,
            epochs=training_info['epochs'],
            batch_size=training_info['batch_size'],
            learning_rate=training_info['learning_rate'],
            output_dir=str(experiment_dir)
        )
        
        # Training plots are already created in train_model function
        print(f"✅ Training completed! Results saved to {experiment_dir}")
    
    if args.mode in ['evaluate', 'train_evaluate']:
        # Determine model path
        if args.model_path:
            model_path = Path(args.model_path)
        else:
            # Resolve latest timestamped train dir for the base experiment name
            base_name = config['output']['experiment_name']
            latest_dir = find_latest_train_experiment_dir(base_name)
            if latest_dir is None:
                # Fallback: create a fresh structure to avoid crashes
                latest_dir = get_experiment_output_dir(config)
            models_dir = latest_dir / "models"
            model_path = models_dir / "best_model.pth"
            
            if not model_path.exists():
                model_path = models_dir / "final_model.pth"
        
        if not model_path.exists():
            print(f"❌ Model not found at {model_path}")
            print("Please provide a valid model path or ensure training completed successfully.")
            return
        
        # For train_evaluate mode, use the actual train experiment name for test
        if args.mode == 'train_evaluate':
            # Extract the actual train experiment name (timestamped) from model path
            model_path_obj = Path(model_path)
            actual_train_experiment_name = model_path_obj.parent.parent.name
            
            # Update config to use the base name but fix the test subdir to the actual timestamped name
            # Keep base name for directory nesting, but ensure subdir matches training run
            config['output']['experiment_name'] = model_path_obj.parent.parent.parent.name
            config['_fixed_test_subdir_name'] = actual_train_experiment_name
        
        print(f"\n🔍 Evaluating model from {model_path}...")
        results = evaluate_model(
            model_path=str(model_path),
            config=config,
            batch_size=training_info['batch_size']
        )
    if args.mode == 'tune':
        # Lazy import to avoid optuna dependency for non-tune runs
        try:
            import importlib
            optuna = importlib.import_module('optuna')
            pruners_mod = importlib.import_module('optuna.pruners')
            MedianPruner = getattr(pruners_mod, 'MedianPruner')
        except Exception as e:
            print("❌ Optuna is not installed. Install it with: pip install optuna")
            return

        base_name = config['output']['experiment_name']
        print(f"\n🧪 Starting Optuna tuning session: {experiment_dir}")

        # Define objective
        def objective(trial):
            # Sample hyperparameters
            sampled_lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
            sampled_batch = trial.suggest_categorical('batch_size', [8, 16, 32, 64])
            # Optionally tune epochs for quicker feedback (or use provided tune_epochs)
            use_epochs = args.tune_epochs if args.tune_epochs is not None else training_info['epochs']

            # Create per-trial output dir
            trial_dir = experiment_dir / f"trial_{trial.number:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # Clone config shallowly and override training params for the trial
            trial_config = dict(config)
            trial_config['training'] = dict(config['training'])
            trial_config['training']['learning_rate'] = sampled_lr
            trial_config['training']['batch_size'] = sampled_batch

            # Build model per trial
            model = get_model(model_info['name'], dataset_info['num_classes'], model_info['pretrained'])

            # Train and report
            try:
                _ = train_model(
                    model=model,
                    config=trial_config,
                    epochs=use_epochs,
                    batch_size=sampled_batch,
                    learning_rate=sampled_lr,
                    output_dir=str(trial_dir),
                    optuna_trial=trial,
                    enable_pruning=args.tune_pruning,
                )
            except Exception as e:
                # In case of out-of-memory or other runtime errors, fail this trial gracefully
                print(f"Trial {trial.number} failed: {e}")
                return 0.0

            # Read best validation accuracy from summary
            summary_path = trial_dir / 'experiment_summary.txt'
            best_val = None
            if summary_path.exists():
                try:
                    with open(summary_path, 'r') as f:
                        for line in f:
                            if line.strip().startswith('Best Validation Accuracy:'):
                                # e.g., 'Best Validation Accuracy: 92.34%'
                                perc = line.split(':')[-1].strip().rstrip('%')
                                best_val = float(perc)
                                break
                except Exception:
                    best_val = None

            # Fallback: parse best from logs if not found
            if best_val is None:
                try:
                    import json
                    with open(trial_dir / 'logs' / 'training_history.json', 'r') as f:
                        hist = json.load(f)
                    best_val = float(max(hist.get('val_acc', [0.0])))
                except Exception:
                    best_val = 0.0

            print(f"   Trial {trial.number} result: best val acc = {best_val:.2f}")
            return best_val

        study = optuna.create_study(direction=args.tune_direction, pruner=MedianPruner() if args.tune_pruning else None)
        study.optimize(objective, n_trials=args.trials)

        # Save study results
        best_params = study.best_params
        best_value = study.best_value
        results_file = experiment_dir / 'tuning_results.txt'
        with open(results_file, 'w') as f:
            f.write("Optuna Tuning Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Direction: {args.tune_direction}\n")
            f.write(f"Trials: {args.trials}\n")
            f.write(f"Best Value (Val Acc): {best_value:.4f}\n")
            f.write(f"Best Params: {best_params}\n")

        # Also dump as JSON
        try:
            import json
            with open(experiment_dir / 'tuning_results.json', 'w') as f:
                json.dump({
                    'direction': args.tune_direction,
                    'trials': args.trials,
                    'best_value': best_value,
                    'best_params': best_params,
                }, f, indent=2)
        except Exception:
            pass

        print(f"✅ Tuning completed. Best value={best_value:.4f}, params={best_params}")

        # Final train_evaluate with best params
        print("\n🏁 Running final train_evaluate with best hyperparameters...")

        # Prepare final config
        final_config = dict(config)
        final_config['training'] = dict(config['training'])
        final_config['training']['learning_rate'] = best_params.get('learning_rate', training_info['learning_rate'])
        final_config['training']['batch_size'] = best_params.get('batch_size', training_info['batch_size'])

        # Create a fresh experiment directory for final training
        final_experiment_dir = get_experiment_output_dir(final_config)

        # Build model
        final_model = get_model(model_info['name'], dataset_info['num_classes'], model_info['pretrained'])

        # Train with early stopping and 300 epochs (default) using best hyperparameters
        _ = train_model(
            model=final_model,
            config=final_config,
            epochs=args.tune_final_epochs,
            batch_size=final_config['training']['batch_size'],
            learning_rate=final_config['training']['learning_rate'],
            output_dir=str(final_experiment_dir),
            early_stopping=True,
            es_patience=args.tune_es_patience,
            es_min_delta=args.tune_es_min_delta,
        )

        # Locate best model path and evaluate
        models_dir = final_experiment_dir / 'models'
        best_model_path = models_dir / 'best_model.pth'
        if not best_model_path.exists():
            best_model_path = models_dir / 'final_model.pth'

        print(f"\n🔍 Evaluating best model from final training: {best_model_path}")
        _ = evaluate_model(
            model_path=str(best_model_path),
            config=final_config,
            batch_size=final_config['training']['batch_size']
        )
        
        # Evaluation plots are already created in evaluate_model function
        print(f"✅ Evaluation completed! Results saved to {experiment_dir}")
    
    # Print final summary
    print(f"\n🎉 Experiment completed successfully!")
    print(f"📁 All results saved to: {experiment_dir}")
    print(f"   📊 Models: {experiment_dir / 'models'}")
    print(f"   📈 Plots: {experiment_dir / 'plots'}")
    print(f"   📋 Results: {experiment_dir / 'results'}")
    print(f"   📝 Logs: {experiment_dir / 'logs'}")
    print(f"   📄 Summary: {experiment_dir / 'experiment_summary.txt'}")


if __name__ == "__main__":
    main() 