"""
Training utilities for classification models
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, List, Tuple, Optional
import json
from pathlib import Path
import time
from tqdm import tqdm

from utils.dataset import create_data_loaders
from utils.config import get_experiment_output_dir
from utils.plot import plot_training_history, plot_learning_curves


def train_model(
    model,
    config: Dict[str, Any],
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    output_dir: Optional[str] = None,
    optuna_trial: Optional[object] = None,
    enable_pruning: bool = False,
    early_stopping: bool = False,
    es_patience: int = 10,
    es_min_delta: float = 0.0,
):
    """
    Train a classification model
    
    Args:
        model: PyTorch model
        config: Configuration dictionary
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        output_dir: Output directory
    
    Returns:
        history: Training history dictionary
    """
    # Get experiment output directory (allow overriding from caller)
    if output_dir is not None:
        experiment_dir = Path(output_dir)
        experiment_dir.mkdir(parents=True, exist_ok=True)
    else:
        experiment_dir = get_experiment_output_dir(config)
    
    # Create subdirectories
    models_dir = experiment_dir / "models"
    plots_dir = experiment_dir / "plots"
    logs_dir = experiment_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Training results will be saved to: {experiment_dir}")
    
    # Create data loaders using config
    train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
        config, batch_size=batch_size
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 클래스별 가중치 설정
    class_weights = None
    if 'class_weights' in config.get('training', {}):
        weights = config['training']['class_weights']
        if isinstance(weights, list) and len(weights) == num_classes:
            class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
            print(f"📊 클래스별 가중치 적용: {dict(zip(class_names, weights))}")
        else:
            print(f"⚠️ 클래스별 가중치 설정이 잘못되었습니다. 무시합니다.")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    # Some torch versions do not support the 'verbose' argument
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    epochs_without_improve = 0
    stopped_early = False
    
    print(f"🚀 Starting training for {epochs} epochs...")
    print(f"   Model: {config['model']['name']}")
    print(f"   Classes: {class_names}")
    print(f"   Device: {device}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Class weights: {class_weights.tolist() if class_weights is not None else 'None'}")
    print(f"   Train samples: {len(train_loader.dataset)}")
    print(f"   Val samples: {len(val_loader.dataset)}")
    print("-" * 60)
    
    # Training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", 
                         leave=False, ncols=100)
        
        for data, target in train_pbar:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                
                # Update progress bar
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print detailed epoch results
        print(f"\n📊 Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc + es_min_delta:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'history': history,
                'classes': class_names,
                'config': config
            }, models_dir / 'best_model.pth')
            print(f"   💾 Best model saved! (Val Acc: {val_acc:.2f}%)")
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

        # Early stopping
        if early_stopping and epochs_without_improve >= es_patience:
            print(f"   ⛔ Early stopping triggered (patience={es_patience}).")
            stopped_early = True
            break

        print("-" * 60)

        # Optuna reporting and pruning
        if optuna_trial is not None:
            try:
                optuna_trial.report(val_acc, step=epoch)
                if enable_pruning and optuna_trial.should_prune():
                    # Import lazily to avoid static import warnings
                    import importlib
                    exceptions_mod = importlib.import_module('optuna.exceptions')
                    TrialPruned = getattr(exceptions_mod, 'TrialPruned')
                    print("   🔪 Trial pruned by Optuna based on intermediate value.")
                    raise TrialPruned()
            except Exception as _e:
                # If optuna is not available or any optuna-related issue occurs, ignore pruning/reporting
                pass
    
    # Calculate total training time
    # Note: epoch_time holds last epoch duration; recompute approximate total based on history length
    total_epochs_run = len(history['val_acc'])
    total_time = time.time() - epoch_start_time + (total_epochs_run * epoch_time)
    
    # Save final model
    torch.save({
        'epoch': total_epochs_run,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history,
        'classes': class_names,
        'config': config
    }, models_dir / 'final_model.pth')
    
    # Save training history
    history_file = logs_dir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Create plots
    plot_training_history(history, plots_dir)
    plot_learning_curves(history, plots_dir)
    
    # Save experiment summary
    summary_file = experiment_dir / 'experiment_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Experiment Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {config['model']['name']}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Epochs Run: {total_epochs_run}\n")
        f.write(f"Early Stopping: {early_stopping}\n")
        if early_stopping:
            f.write(f"ES Patience: {es_patience}\n")
            f.write(f"ES Min Delta: {es_min_delta}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Class Weights: {class_weights.tolist() if class_weights is not None else 'None'}\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Validation Accuracy: {val_acc:.2f}%\n")
        f.write(f"Transform Type: {config.get('dataset', {}).get('augmentation', {}).get('transform_type', 'standard')}\n")
        f.write(f"Total Training Time: {total_time/60:.1f} minutes\n")
        f.write(f"Average Epoch Time: {epoch_time:.1f} seconds\n")
    
    # Save experiment specification
    spec_file = experiment_dir / 'experiment_specification.txt'
    with open(spec_file, 'w') as f:
        f.write("Experiment Specification\n")
        f.write("=" * 50 + "\n")
        f.write(f"Experiment Name: {config['output']['experiment_name']}\n")
        f.write(f"Experiment Type: Training\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config File: {config.get('_config_file_path', 'Unknown')}\n\n")
        
        f.write("Dataset Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Train File: {config['dataset']['train_txt']}\n")
        f.write(f"Validation File: {config['dataset']['val_txt']}\n")
        f.write(f"Test File: {config['dataset']['test_txt']}\n")
        f.write(f"Classes: {config['dataset']['classes']}\n")
        f.write(f"Number of Classes: {len(config['dataset']['classes'])}\n")
        f.write(f"Train Samples: {len(train_loader.dataset)}\n")
        f.write(f"Validation Samples: {len(val_loader.dataset)}\n")
        f.write(f"Test Samples: {len(test_loader.dataset)}\n\n")
        
        f.write("Model Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Name: {config['model']['name']}\n")
        f.write(f"Pretrained: {config['model']['pretrained']}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")
        
        f.write("Training Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Epochs: {config['training']['epochs']}\n")
        f.write(f"Batch Size: {config['training']['batch_size']}\n")
        f.write(f"Learning Rate: {config['training']['learning_rate']}\n")
        f.write(f"Number of Workers: {config['training']['num_workers']}\n")
        f.write(f"Class Weights: {config['training'].get('class_weights', 'None')}\n\n")
        
        f.write("Data Augmentation:\n")
        f.write("-" * 30 + "\n")
        augmentation = config.get('dataset', {}).get('augmentation', {})
        f.write(f"Transform Type: {augmentation.get('transform_type', 'standard')}\n")
        f.write(f"Resize: {augmentation.get('resize', [224, 224])}\n")
        f.write(f"Random Horizontal Flip: {augmentation.get('random_horizontal_flip', 0.5)}\n")
        f.write(f"Random Rotation: {augmentation.get('random_rotation', 10)}\n")
        f.write(f"Color Jitter: {augmentation.get('color_jitter', {})}\n")
        f.write(f"Normalize: {augmentation.get('normalize', {})}\n\n")
        
        f.write("Output Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Experiment Name: {config['output']['experiment_name']}\n")
        f.write(f"Exist OK: {config['output']['exist_ok']}\n")
        f.write(f"Save Best Model: {config['output']['save_best_model']}\n")
        f.write(f"Save Training History: {config['output']['save_training_history']}\n\n")
        
        f.write("Training Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Validation Accuracy: {val_acc:.2f}%\n")
        f.write(f"Total Training Time: {total_time/60:.1f} minutes\n")
        f.write(f"Average Epoch Time: {epoch_time:.1f} seconds\n")
        f.write(f"Final Train Loss: {avg_train_loss:.4f}\n")
        f.write(f"Final Train Accuracy: {train_acc:.2f}%\n")
        f.write(f"Final Val Loss: {avg_val_loss:.4f}\n")
        f.write(f"Final Val Accuracy: {val_acc:.2f}%\n")
    
    print(f"\n✅ Training completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print(f"📁 Results saved to: {experiment_dir}")
    print(f"   Models: {models_dir}")
    print(f"   Plots: {plots_dir}")
    print(f"   Logs: {logs_dir}")
    print(f"   Summary: {summary_file}")
    print(f"   Specification: {spec_file}")
    
    return history


if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Train classification model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    
    args = parser.parse_args()
    
    try:
        from utils.config import load_config, get_training_info
        from models.resnet import ResNet18, ResNet50
        from models.efficientnet import EfficientNet
        from models.mobilenet import MobileNet
        
        # Load configuration
        config = load_config(args.config)
        
        # Get training info
        training_info = get_training_info(config)
        
        # Override settings if provided
        if args.epochs:
            training_info['epochs'] = args.epochs
        if args.batch_size:
            training_info['batch_size'] = args.batch_size
        if args.lr:
            training_info['learning_rate'] = args.lr
        
        # Get model
        model_name = config['model']['name']
        num_classes = len(config['dataset']['classes'])
        pretrained = config['model'].get('pretrained', True)
        
        if model_name == 'resnet18':
            model = ResNet18(num_classes=num_classes, pretrained=pretrained)
        elif model_name == 'resnet50':
            model = ResNet50(num_classes=num_classes, pretrained=pretrained)
        elif model_name == 'efficientnet':
            model = EfficientNet(num_classes=num_classes, pretrained=pretrained)
        elif model_name == 'mobilenet':
            model = MobileNet(num_classes=num_classes, pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Train model
        history = train_model(
            model=model,
            config=config,
            epochs=training_info['epochs'],
            batch_size=training_info['batch_size'],
            learning_rate=training_info['learning_rate']
        )
        
        print(f"🎉 Training completed successfully!")
        
    except ImportError:
        print("Error: Could not import required modules. Make sure you're running from the correct directory.")
        print("Try running: python -m train --config config.yaml")
    except Exception as e:
        print(f"Error during training: {e}") 