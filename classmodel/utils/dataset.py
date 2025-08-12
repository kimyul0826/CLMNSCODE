"""
Dataset utilities for classification
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any

from .transforms import get_transforms_by_split_config


class CustomDataset(Dataset):
    """
    Custom dataset for image classification using train.txt, val.txt, test.txt files
    """
    
    def __init__(self, config: Dict[str, Any], split='train', transform=None):
        """
        Initialize the dataset
        
        Args:
            config: Configuration dictionary containing dataset info
            split: 'train', 'val', or 'test'
            transform: Image transformations
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.classes = []
        self.class_to_idx = {}
        self.samples = []
        
        # Get dataset info from config
        dataset_info = get_dataset_info(config)
        
        # Define the txt file for this split
        txt_file_map = {
            'train': dataset_info['train_txt'],
            'val': dataset_info['val_txt'],
            'test': dataset_info['test_txt']
        }
        
        txt_path = txt_file_map[split]
        
        if not os.path.exists(txt_path):
            raise ValueError(f"File {txt_path} not found")
        
        # Read samples from the txt file
        all_samples = []
        class_indices = set()
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:  # Skip empty lines
                # Format: image_path class_index
                parts = line.split()
                if len(parts) >= 2:
                    image_path = parts[0]
                    class_index = int(parts[1])  # Convert to integer
                    
                    # Add class index to set
                    class_indices.add(class_index)
                    
                    # Store sample
                    all_samples.append((image_path, class_index))
                else:
                    print(f"Warning: Skipping invalid line: {line}")
        
        # Create class mapping (class indices to consecutive indices)
        class_indices = sorted(list(class_indices))
        for idx, class_idx in enumerate(class_indices):
            self.class_to_idx[class_idx] = idx
            # Use class names from config if available
            class_name = dataset_info['classes'].get(class_idx, f"Class_{class_idx}")
            self.classes.append(class_name)
        
        # Convert class indices to consecutive indices
        for image_path, class_index in all_samples:
            if class_index in self.class_to_idx:
                label = self.class_to_idx[class_index]
                self.samples.append((image_path, label))
            else:
                print(f"Warning: Unknown class index '{class_index}' for image {image_path}")

        # Expose original image paths aligned with dataset order for downstream utilities
        # This allows evaluation utilities to retrieve and save the actual source images
        # corresponding to predictions without relying on placeholders.
        self.image_paths = [image_path for image_path, _ in self.samples]
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Classes: {self.classes}")
        print(f"Class distribution:")
        for class_name in self.classes:
            class_idx = self.classes.index(class_name)
            count = sum(1 for _, label in self.samples if label == class_idx)
            print(f"  {class_name}: {count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_weights(self):
        """
        Calculate class weights for imbalanced datasets
        
        Returns:
            class_weights: Tensor of class weights
        """
        class_counts = np.zeros(len(self.classes))
        
        for _, label in self.samples:
            class_counts[label] += 1
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.samples)
        class_weights = total_samples / (len(self.classes) * class_counts)
        
        return torch.FloatTensor(class_weights)


def create_data_loaders(config: Dict[str, Any], batch_size=None, num_workers=None):
    """
    Create train, validation, and test data loaders
    
    Args:
        config: Configuration dictionary
        batch_size: Override batch size from config
        num_workers: Override num_workers from config
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, class_names
    """
    from torch.utils.data import DataLoader
    from utils.config import get_training_info
    
    # Get training info from config
    training_info = get_training_info(config)
    dataset_info = get_dataset_info(config)
    
    # Use provided parameters or config values
    batch_size = batch_size or training_info['batch_size']
    num_workers = num_workers or training_info['num_workers']
    
    # Get transforms from split-specific config
    transforms_dict = get_transforms_by_split_config(config)
    
    # Print transform information
    print(f"📊 Transform Configuration:")
    for split, transform in transforms_dict.items():
        print(f"   {split}: {transform}")
    
    # Create datasets
    train_dataset = CustomDataset(config, split='train', transform=transforms_dict['train'])
    val_dataset = CustomDataset(config, split='val', transform=transforms_dict['val'])
    test_dataset = CustomDataset(config, split='test', transform=transforms_dict['test'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    num_classes = len(train_dataset.classes)
    class_names = train_dataset.classes
    
    return train_loader, val_loader, test_loader, num_classes, class_names


def get_dataset_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get dataset information from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        dataset_info: Dictionary with dataset information
    """
    dataset = config['dataset']
    
    # Count number of classes from classes dict
    num_classes = len(dataset['classes'])
    
    # Update num_classes in config if it doesn't match
    if dataset.get('num_classes', 0) != num_classes:
        dataset['num_classes'] = num_classes
        print(f"Updated num_classes to {num_classes} based on classes dictionary")
    
    return {
        'train_txt': dataset['train_txt'],
        'val_txt': dataset['val_txt'],
        'test_txt': dataset['test_txt'],
        'classes': dataset['classes'],
        'num_classes': num_classes,
        'augmentation': dataset.get('augmentation', {})
    }


def validate_dataset(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the dataset and check for missing images
    
    Args:
        config: Configuration dictionary
    
    Returns:
        validation_results: Dictionary with validation results
    """
    dataset_info = get_dataset_info(config)
    splits = ['train', 'val', 'test']
    txt_files = [dataset_info['train_txt'], dataset_info['val_txt'], dataset_info['test_txt']]
    
    validation_results = {
        'total_images': 0,
        'valid_images': 0,
        'missing_images': 0,
        'missing_paths': [],
        'splits': {}
    }
    
    for split, txt_path in zip(splits, txt_files):
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found")
            continue
        
        split_results = {
            'images': 0,
            'valid_images': 0,
            'missing_images': 0,
            'missing_paths': []
        }
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    image_path = parts[0]
                    class_index = int(parts[1])
                    
                    split_results['images'] += 1
                    validation_results['total_images'] += 1
                    
                    if os.path.exists(image_path):
                        split_results['valid_images'] += 1
                        validation_results['valid_images'] += 1
                    else:
                        split_results['missing_images'] += 1
                        validation_results['missing_images'] += 1
                        split_results['missing_paths'].append(image_path)
                        validation_results['missing_paths'].append(image_path)
        
        validation_results['splits'][split] = split_results
    
    return validation_results


if __name__ == "__main__":
    # Example usage
    import argparse
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Dataset utilities')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--validate', action='store_true',
                       help='Validate dataset and check for missing images')
    
    args = parser.parse_args()
    
    try:
        from utils.config import load_config
        
        # Load configuration
        config = load_config(args.config)
        
        # Get dataset info
        info = get_dataset_info(config)
        print("Dataset Information:")
        print(f"Number of classes: {info['num_classes']}")
        print(f"Classes: {info['classes']}")
        print(f"Train txt: {info['train_txt']}")
        print(f"Val txt: {info['val_txt']}")
        print(f"Test txt: {info['test_txt']}")
        
        # Validate dataset if requested
        if args.validate:
            print("\nValidating dataset...")
            validation_results = validate_dataset(config)
            print(f"Total images: {validation_results['total_images']}")
            print(f"Valid images: {validation_results['valid_images']}")
            print(f"Missing images: {validation_results['missing_images']}")
            
            for split, split_results in validation_results['splits'].items():
                print(f"\n{split} split:")
                print(f"  Images: {split_results['images']}")
                print(f"  Valid: {split_results['valid_images']}")
                print(f"  Missing: {split_results['missing_images']}")
                
                if split_results['missing_images'] > 0:
                    print(f"  Missing image paths:")
                    for path in split_results['missing_paths'][:5]:  # Show first 5
                        print(f"    {path}")
                    if len(split_results['missing_paths']) > 5:
                        print(f"    ... and {len(split_results['missing_paths']) - 5} more")
        
        # Create data loaders
        try:
            train_loader, val_loader, test_loader, num_classes, class_names = create_data_loaders(
                config, batch_size=args.batch_size
            )
            
            print(f"\nData loaders created:")
            print(f"Train batches: {len(train_loader)}")
            print(f"Val batches: {len(val_loader)}")
            print(f"Test batches: {len(test_loader)}")
            print(f"Classes: {class_names}")
        except Exception as e:
            print(f"Error creating data loaders: {e}")
            print("Please check your configuration and txt files.")
    
    except ImportError:
        print("Error: Could not import config module. Make sure you're running from the correct directory.")
        print("Try running: python -m utils.dataset --config config.yaml")
    except Exception as e:
        print(f"Error: {e}") 