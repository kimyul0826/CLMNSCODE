"""
Plotting utilities for training history and evaluation results
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd


def plot_training_history(history, output_dir):
    """
    Plot training history (loss and accuracy)
    
    Args:
        history: Training history dictionary
        output_dir: Output directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {output_dir}/training_history.png")


def plot_confusion_matrix(conf_matrix, class_names, output_dir):
    """
    Plot confusion matrix
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names
        output_dir: Output directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Create heatmap
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix plot saved to {output_dir}/confusion_matrix.png")


def plot_class_accuracy(per_class_acc, class_names, output_dir):
    """
    Plot per-class accuracy
    
    Args:
        per_class_acc: Per-class accuracy array
        class_names: List of class names
        output_dir: Output directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_acc)
    
    # Add value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.title('Per-Class Accuracy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Per-class accuracy plot saved to {output_dir}/per_class_accuracy.png")


def plot_learning_curves(history, output_dir):
    """
    Plot detailed learning curves
    
    Args:
        history: Training history dictionary
        output_dir: Output directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Validation loss
    ax2.plot(epochs, history['val_loss'], 'r-', linewidth=2)
    ax2.set_title('Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Training accuracy
    ax3.plot(epochs, history['train_acc'], 'b-', linewidth=2)
    ax3.set_title('Training Accuracy')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.grid(True, alpha=0.3)
    
    # Validation accuracy
    ax4.plot(epochs, history['val_acc'], 'r-', linewidth=2)
    ax4.set_title('Validation Accuracy')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Learning curves plot saved to {output_dir}/learning_curves.png")


def plot_metrics_comparison(metrics_dict, output_dir):
    """
    Plot comparison of different metrics
    
    Args:
        metrics_dict: Dictionary with model names as keys and metrics as values
        output_dir: Output directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[model]['accuracy'] for model in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model comparison plot saved to {output_dir}/model_comparison.png")


def create_summary_report(history, results, output_dir):
    """
    Create a comprehensive summary report
    
    Args:
        history: Training history dictionary
        results: Evaluation results dictionary
        output_dir: Output directory for saving report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary text
    summary = f"""
# Model Training and Evaluation Summary

## Training Results
- Final Training Loss: {history['train_loss'][-1]:.4f}
- Final Training Accuracy: {history['train_acc'][-1]:.2f}%
- Final Validation Loss: {history['val_loss'][-1]:.4f}
- Final Validation Accuracy: {history['val_acc'][-1]:.2f}%
- Best Validation Accuracy: {max(history['val_acc']):.2f}%

## Evaluation Results
- Overall Accuracy: {results['accuracy']:.4f}
- Per-class Accuracy:
"""
    
    for i, acc in enumerate(results['per_class_accuracy']):
        summary += f"  - {results['class_names'][i]}: {acc:.4f}\n"
    
    summary += f"""
## Model Architecture
- Number of Classes: {len(results['class_names'])}
- Classes: {', '.join(results['class_names'])}

## Files Generated
- Training History Plot: training_history.png
- Confusion Matrix: confusion_matrix.png
- Per-class Accuracy: per_class_accuracy.png
- Learning Curves: learning_curves.png
"""
    
    # Save summary
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write(summary)
    
    print(f"Summary report saved to {output_dir}/summary_report.txt")

# Grid search plotting helpers removed


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot training and evaluation results')
    parser.add_argument('--history_file', type=str, help='Path to history JSON file')
    parser.add_argument('--results_file', type=str, help='Path to results JSON file')
    parser.add_argument('--output_dir', type=str, default='./plots', help='Output directory')
    
    args = parser.parse_args()
    
    if args.history_file and os.path.exists(args.history_file):
        import json
        with open(args.history_file, 'r') as f:
            history = json.load(f)
        plot_training_history(history, args.output_dir)
        plot_learning_curves(history, args.output_dir)
    
    if args.results_file and os.path.exists(args.results_file):
        import json
        with open(args.results_file, 'r') as f:
            results = json.load(f)
        plot_confusion_matrix(results['confusion_matrix'], results['class_names'], args.output_dir)
        plot_class_accuracy(results['per_class_accuracy'], results['class_names'], args.output_dir) 