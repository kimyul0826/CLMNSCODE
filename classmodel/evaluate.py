"""
Evaluation utilities for classification models
"""

import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from typing import Dict, Any, List, Tuple
import json
from pathlib import Path
import time
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.dataset import create_data_loaders
from utils.config import get_test_experiment_output_dir
from utils.plot import plot_confusion_matrix, plot_class_accuracy, create_summary_report


def save_prediction_images(test_loader, all_predictions, all_targets, class_names, output_dir):
    """
    Save images with prediction results
    
    Args:
        test_loader: Test data loader
        all_predictions: List of predictions
        all_targets: List of true targets
        class_names: List of class names
        output_dir: Output directory for saving images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📸 Saving prediction images...")
    
    # Get original images from test dataset
    test_dataset = test_loader.dataset
    
    # Create subdirectories for correct and incorrect predictions
    correct_dir = output_dir / "correct"
    incorrect_dir = output_dir / "incorrect"
    correct_dir.mkdir(exist_ok=True)
    incorrect_dir.mkdir(exist_ok=True)
    
    correct_count = 0
    incorrect_count = 0
    
    for idx, (pred, true) in enumerate(zip(all_predictions, all_targets)):
        # Get original image path and image
        if hasattr(test_dataset, 'image_paths'):
            # CustomDataset with image paths
            img_path = test_dataset.image_paths[idx]
            img_name = Path(img_path).name
        else:
            # Fallback for other dataset types
            img_name = f"sample_{idx:04d}.jpg"
        
        # Load and process image
        if hasattr(test_dataset, 'image_paths'):
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                # If image path doesn't exist, create a placeholder
                img = Image.new('RGB', (224, 224), color='gray')
        else:
            # Create placeholder image
            img = Image.new('RGB', (224, 224), color='gray')
        
        # Resize image to standard size
        img = img.resize((224, 224))
        
        # Create a copy for drawing
        img_with_text = img.copy()
        draw = ImageDraw.Draw(img_with_text)
        
        # Try to load font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Prepare text
        true_class = class_names[true]
        pred_class = class_names[pred]
        is_correct = pred == true

        # Text content (no background box)
        text = f"Actual : {true_class} , Predict : {pred_class}"

        # Text color based on correctness
        if is_correct:
            text_color = (0, 255, 0)  # Green for correct
            save_dir = correct_dir
            correct_count += 1
        else:
            text_color = (255, 0, 0)  # Red for incorrect
            save_dir = incorrect_dir
            incorrect_count += 1

        # Draw text directly without background
        draw.text((5, 5), text, fill=text_color, font=font)
        
        # Save image
        save_path = save_dir / f"{img_name}"
        img_with_text.save(save_path)
    
    print(f"   ✅ Correct predictions: {correct_count} images saved to {correct_dir}")
    print(f"   ❌ Incorrect predictions: {incorrect_count} images saved to {incorrect_dir}")
    print(f"   📊 Total images: {len(all_predictions)}")
    print(f"   📈 Accuracy: {correct_count}/{len(all_predictions)} ({correct_count/len(all_predictions)*100:.1f}%)")


def evaluate_model(model_path, config: Dict[str, Any], batch_size=32):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to the trained model
        config: Configuration dictionary
        batch_size: Batch size for evaluation
    
    Returns:
        results: Evaluation results dictionary
    """
    # Extract base and timestamped train experiment names from model path
    # Support both nested: runs/train/{base}/{base_YYMMDD_HHMMSS}/models/best_model.pth
    # and flat:   runs/train/{base_YYMMDD_HHMMSS}/models/best_model.pth
    model_path_obj = Path(model_path)
    timestamped_train_run = model_path_obj.parent.parent.name

    # Try to parse base from timestamped folder name if needed
    def _derive_base_from_timestamped(name: str) -> str:
        parts = name.split('_')
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit() and len(parts[-1]) == 6 and len(parts[-2]) == 6:
            return '_'.join(parts[:-2]) if len(parts[:-2]) > 0 else name
        return name

    parent3_name = model_path_obj.parent.parent.parent.name
    if parent3_name in {"train", "runs", ""}:
        base_experiment_name = _derive_base_from_timestamped(timestamped_train_run)
    else:
        base_experiment_name = parent3_name

    # Update config to nest test outputs as runs/test/{base}/{timestamped}
    config['output']['experiment_name'] = base_experiment_name
    config['_fixed_test_subdir_name'] = timestamped_train_run
    
    # Get test experiment output directory
    experiment_dir = get_test_experiment_output_dir(config)
    
    # Create subdirectories
    plots_dir = experiment_dir / "plots"
    logs_dir = experiment_dir / "logs"
    results_dir = experiment_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    print(f"📁 Evaluation results will be saved to: {experiment_dir}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    model_state_dict = checkpoint['model_state_dict']
    classes = checkpoint.get('classes', ['Class_0', 'Class_1'])
    
    # Get model architecture
    from utils.config import get_model_info
    model_info = get_model_info(config)
    model_name = model_info['name']
    num_classes = len(classes)
    pretrained = model_info.get('pretrained', True)
    
    # Create model
    if model_name == 'resnet18':
        from models.resnet import ResNet18
        model = ResNet18(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'resnet50':
        from models.resnet import ResNet50
        model = ResNet50(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'efficientnet':
        from models.efficientnet import EfficientNet
        model = EfficientNet(num_classes=num_classes, pretrained=pretrained)
    elif model_name == 'mobilenet':
        from models.mobilenet import MobileNet
        model = MobileNet(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    print(f"🔍 Evaluating model: {model_name}")
    print(f"   Classes: {classes}")
    print(f"   Device: {device}")
    print(f"   Model path: {model_path}")
    print("-" * 50)
    
    # Create test data loader
    train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
        config, batch_size=batch_size
    )
    
    print(f"📊 Test samples: {len(test_loader.dataset)}")
    print(f"📊 Test batches: {len(test_loader)}")
    print("-" * 50)
    
    # Start evaluation timer
    eval_start_time = time.time()
    
    # Evaluation
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    # Progress bar for evaluation
    test_pbar = tqdm(test_loader, desc="Evaluating", ncols=100)
    
    with torch.no_grad():
        for data, target in test_pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Update progress bar
            test_pbar.set_postfix({
                'Batch': f'{len(all_predictions)}/{len(test_loader.dataset)}'
            })
    
    # Calculate evaluation time
    eval_time = time.time() - eval_start_time
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = np.mean(all_predictions == all_targets)
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    # Per-class accuracy
    per_class_accuracy = []
    for i in range(num_classes):
        class_mask = all_targets == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(all_predictions[class_mask] == all_targets[class_mask])
        else:
            class_acc = 0.0
        per_class_accuracy.append(class_acc)
    
    # Calculate precision, recall, f1-score, support
    precision, recall, f1_score, support = precision_recall_fscore_support(
        all_targets, all_predictions, average=None, zero_division=0
    )
    
    # Classification report
    report = classification_report(all_targets, all_predictions, 
                                 target_names=class_names, output_dict=True)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'per_class_accuracy': per_class_accuracy,
        'class_names': class_names,
        'predictions': all_predictions.tolist(),
        'targets': all_targets.tolist(),
        'probabilities': all_probabilities.tolist(),
        'classification_report': report
    }
    
    # Print results
    print(f"\n📊 Evaluation Results ({eval_time:.1f}s):")
    print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Per-class Accuracy:")
    for i, (class_name, acc) in enumerate(zip(class_names, per_class_accuracy)):
        print(f"     {class_name}: {acc:.4f} ({acc*100:.2f}%)")
    
    # Print detailed metrics
    print(f"\n📈 Detailed Metrics:")
    print(f"   {'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 55)
    for i, class_name in enumerate(class_names):
        print(f"   {class_name:<12} {precision[i]:<10.4f} {recall[i]:<10.4f} {f1_score[i]:<10.4f} {support[i]:<10.0f}")
    
    # Print confusion matrix (count)
    print(f"\n📋 Confusion Matrix (Count):")
    print("   Predicted")
    print("   " + " " * 12 + " ".join([f"{name:>8}" for name in class_names]))
    for i, true_class in enumerate(class_names):
        row = f"   {true_class:<12}"
        for j in range(num_classes):
            count = conf_matrix[i, j]
            row += f"{count:>8}"
        print(row)
    
    # Print confusion matrix (ratio)
    print(f"\n📋 Confusion Matrix (Ratio):")
    print("   Predicted")
    print("   " + " " * 12 + " ".join([f"{name:>8}" for name in class_names]))
    for i, true_class in enumerate(class_names):
        row = f"   {true_class:<12}"
        total = np.sum(conf_matrix[i, :])
        for j in range(num_classes):
            if total > 0:
                ratio = conf_matrix[i, j] / total
                row += f"{ratio:>7.2f}"
            else:
                row += f"{0:>7.2f}"
        print(row)
    
    # Save prediction images
    prediction_images_dir = results_dir / "prediction_images"
    save_prediction_images(test_loader, all_predictions, all_targets, class_names, prediction_images_dir)
    
    # Save results
    results_file = results_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    plot_confusion_matrix(conf_matrix, class_names, plots_dir)
    plot_class_accuracy(per_class_accuracy, class_names, plots_dir)
    
    # Save detailed report
    report_file = results_dir / 'classification_report.txt'
    with open(report_file, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Evaluation Time: {eval_time:.1f} seconds\n\n")
        
        f.write("Per-class Accuracy:\n")
        for i, (class_name, acc) in enumerate(zip(class_names, per_class_accuracy)):
            f.write(f"  {class_name}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        f.write("\nConfusion Matrix:\n")
        f.write(str(conf_matrix))
        
        f.write("\n\nDetailed Classification Report:\n")
        f.write(classification_report(all_targets, all_predictions, target_names=class_names))
    
    # Save experiment summary
    summary_file = experiment_dir / 'evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Evaluation Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        # Handle both old (dataset.augmentation) and new (dataset.transforms/dataset.resize/normalize) configs
        ds_cfg = config.get('dataset', {})
        aug_cfg = ds_cfg.get('augmentation', {}) if isinstance(ds_cfg.get('augmentation', {}), dict) else {}
        transforms_cfg = ds_cfg.get('transforms', {}) if isinstance(ds_cfg.get('transforms', {}), dict) else {}
        fallback_transform = transforms_cfg.get('test', transforms_cfg.get('val', transforms_cfg.get('train', 'standard')))
        f.write(f"Transform Type: {aug_cfg.get('transform_type', fallback_transform)}\n")
        f.write(f"Test Samples: {len(all_targets)}\n")
        f.write(f"Evaluation Time: {eval_time:.1f} seconds\n")
    
    # Save experiment specification
    spec_file = experiment_dir / 'experiment_specification.txt'
    with open(spec_file, 'w') as f:
        f.write("Experiment Specification\n")
        f.write("=" * 50 + "\n")
        f.write(f"Experiment Name: {config['output']['experiment_name']}\n")
        f.write(f"Experiment Type: Evaluation\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Config File: {config.get('_config_file_path', 'Unknown')}\n\n")
        
        f.write("Dataset Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Train File: {config['dataset']['train_txt']}\n")
        f.write(f"Validation File: {config['dataset']['val_txt']}\n")
        f.write(f"Test File: {config['dataset']['test_txt']}\n")
        f.write(f"Classes: {config['dataset']['classes']}\n")
        f.write(f"Number of Classes: {len(config['dataset']['classes'])}\n")
        f.write(f"Test Samples: {len(all_targets)}\n\n")
        
        f.write("Model Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Pretrained: {config['model']['pretrained']}\n")
        f.write(f"Number of Classes: {num_classes}\n\n")
        
        f.write("Evaluation Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Evaluation Time: {eval_time:.1f} seconds\n\n")
        
        f.write("Data Augmentation:\n")
        f.write("-" * 30 + "\n")
        ds_cfg = config.get('dataset', {})
        aug_cfg = ds_cfg.get('augmentation', {}) if isinstance(ds_cfg.get('augmentation', {}), dict) else {}
        transforms_cfg = ds_cfg.get('transforms', {}) if isinstance(ds_cfg.get('transforms', {}), dict) else {}
        fallback_transform = transforms_cfg.get('test', transforms_cfg.get('val', transforms_cfg.get('train', 'standard')))
        fallback_resize = ds_cfg.get('resize', [224, 224])
        fallback_norm = ds_cfg.get('normalize', {})
        f.write(f"Transform Type: {aug_cfg.get('transform_type', fallback_transform)}\n")
        f.write(f"Resize: {aug_cfg.get('resize', fallback_resize)}\n")
        f.write(f"Random Horizontal Flip: {aug_cfg.get('random_horizontal_flip', 0.5)}\n")
        f.write(f"Random Rotation: {aug_cfg.get('random_rotation', 10)}\n")
        f.write(f"Color Jitter: {aug_cfg.get('color_jitter', {})}\n")
        f.write(f"Normalize: {aug_cfg.get('normalize', fallback_norm)}\n\n")
        
        f.write("Output Configuration:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Experiment Name: {config['output']['experiment_name']}\n")
        f.write(f"Exist OK: {config['output']['exist_ok']}\n")
        f.write(f"Save Best Model: {config['output']['save_best_model']}\n")
        f.write(f"Save Training History: {config['output']['save_training_history']}\n\n")
        
        f.write("Evaluation Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        f.write(f"Evaluation Time: {eval_time:.1f} seconds\n")
        f.write(f"Total Test Samples: {len(all_targets)}\n")
        f.write(f"Correct Predictions: {np.sum(all_predictions == all_targets)}\n")
        f.write(f"Incorrect Predictions: {len(all_predictions) - np.sum(all_predictions == all_targets)}\n\n")
        
        f.write("Per-class Results:\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  {class_name}:\n")
            f.write(f"    Accuracy: {per_class_accuracy[i]:.4f} ({per_class_accuracy[i]*100:.2f}%)\n")
            f.write(f"    Precision: {precision[i]:.4f}\n")
            f.write(f"    Recall: {recall[i]:.4f}\n")
            f.write(f"    F1-Score: {f1_score[i]:.4f}\n")
            f.write(f"    Support: {support[i]:.0f}\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write(str(conf_matrix))
    
    print(f"\n✅ Evaluation completed!")
    print(f"⏱️  Evaluation time: {eval_time:.1f} seconds")
    print(f"📁 Results saved to: {experiment_dir}")
    print(f"   Results: {results_dir}")
    print(f"   Plots: {plots_dir}")
    print(f"   Logs: {logs_dir}")
    print(f"   Summary: {summary_file}")
    print(f"   Specification: {spec_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Evaluate classification model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    try:
        from utils.config import load_config
        
        # Load configuration
        config = load_config(args.config)
        
        # Evaluate model
        results = evaluate_model(
            model_path=args.model_path,
            config=config,
            batch_size=args.batch_size
        )
        
        print(f"🎉 Evaluation completed successfully!")
        
    except ImportError:
        print("Error: Could not import required modules. Make sure you're running from the correct directory.")
        print("Try running: python -m evaluate --config config.yaml --model_path model.pth")
    except Exception as e:
        print(f"Error during evaluation: {e}") 