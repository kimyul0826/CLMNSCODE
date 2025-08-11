"""
Configuration utilities for classification
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import pytz


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        config: Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Add config file path to config for reference
        config['_config_file_path'] = os.path.abspath(config_path)
        
        # Validate configuration
        validate_config(config)
        
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration structure
    
    Args:
        config: Configuration dictionary
    """
    required_sections = ['dataset', 'model', 'training', 'output']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate dataset section
    dataset = config['dataset']
    required_dataset_fields = ['train_txt', 'val_txt', 'test_txt', 'classes']
    for field in required_dataset_fields:
        if field not in dataset:
            raise ValueError(f"Missing required dataset field: {field}")
    
    # Check if txt files exist
    for txt_file in [dataset['train_txt'], dataset['val_txt'], dataset['test_txt']]:
        if not os.path.exists(txt_file):
            print(f"Warning: Text file not found: {txt_file}")
    
    # Validate model section
    model = config['model']
    if 'name' not in model:
        raise ValueError("Missing model name in model section")
    
    valid_models = ['resnet18', 'resnet50', 'efficientnet', 'mobilenet']
    if model['name'] not in valid_models:
        raise ValueError(f"Invalid model name: {model['name']}. Valid options: {valid_models}")
    
    # Validate training section
    training = config['training']
    required_training_fields = ['epochs', 'batch_size', 'learning_rate']
    for field in required_training_fields:
        if field not in training:
            raise ValueError(f"Missing required training field: {field}")
    
    # Validate output section (dir is optional; we manage paths internally)
    output = config['output']
    if 'experiment_name' not in output:
        raise ValueError("Missing experiment_name in output section")


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


def get_model_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get model information from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        model_info: Dictionary with model information
    """
    return config['model']


def get_training_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get training information from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        training_info: Dictionary with training information
    """
    training = config['training']
    
    # Add default num_workers if not specified
    if 'num_workers' not in training:
        training['num_workers'] = 4
    
    return training


def get_output_info(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get output information from configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        output_info: Dictionary with output information
    """
    return config['output']


def _get_kst_timestamp() -> str:
    """
    Return timestamp string in KST as YYMMDD_HHMMSS
    """
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)
    return now_kst.strftime('%y%m%d_%H%M%S')


def _build_timestamped_name(base_name: str) -> str:
    """
    Build experiment name with KST timestamp suffix
    """
    return f"{base_name}_{_get_kst_timestamp()}"


def create_experiment_dir(experiment_name: str, exist_ok: bool = False, force_timestamp: bool = False) -> Path:
    """
    Create experiment directory in runs folder
    
    Args:
        experiment_name: Name of the experiment
        exist_ok: Whether to overwrite existing directory
        force_timestamp: Deprecated, kept for backward compatibility (ignored)
    
    Returns:
        experiment_dir: Path to experiment directory
    """
    # Create runs directory if it doesn't exist
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # Create train and test directories
    train_dir = runs_dir / "train"
    test_dir = runs_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    # Nested structure: runs/train/{base}/{base}_{YYMMDD_HHMMSS}
    base_dir = train_dir / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    timestamped_name = _build_timestamped_name(experiment_name)
    experiment_dir = base_dir / timestamped_name

    if not exist_ok:
        # Extremely unlikely due to timestamp, but keep safe fallback
        counter = 1
        original_name = timestamped_name
        while experiment_dir.exists():
            next_name = f"{original_name}_{counter}"
            experiment_dir = base_dir / next_name
            counter += 1
            if counter > 10:
                break

    # Create experiment directory
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Experiment directory created: {experiment_dir}")
    return experiment_dir


def create_test_experiment_dir(experiment_name: str, exist_ok: bool = False, fixed_subdir: Optional[str] = None) -> Path:
    """
    Create test experiment directory in runs/test folder

    Args:
        experiment_name: Name of the experiment
        exist_ok: Whether to overwrite existing directory

    Returns:
        experiment_dir: Path to test experiment directory
    """
    # Create runs directory if it doesn't exist
    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    
    # Create test directory
    test_dir = runs_dir / "test"
    test_dir.mkdir(exist_ok=True)

    # Nested structure: runs/test/{base}/{base}_{YYMMDD_HHMMSS}
    base_dir = test_dir / experiment_name
    base_dir.mkdir(parents=True, exist_ok=True)

    if fixed_subdir is not None:
        experiment_dir = base_dir / fixed_subdir
    else:
        timestamped_name = _build_timestamped_name(experiment_name)
        experiment_dir = base_dir / timestamped_name

    if not exist_ok:
        counter = 1
        original_path = experiment_dir
        while experiment_dir.exists():
            experiment_dir = base_dir / f"{original_path.name}_{counter}"
            counter += 1
            if counter > 10:
                break

    experiment_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Test experiment directory created: {experiment_dir}")
    return experiment_dir


def get_experiment_output_dir(config: Dict[str, Any], force_timestamp: bool = False) -> Path:
    """
    Get experiment output directory based on config

    Args:
        config: Configuration dictionary
        force_timestamp: Deprecated, kept for backward compatibility (ignored)

    Returns:
        experiment_dir: Path to experiment directory
    """
    output_info = get_output_info(config)
    experiment_name = output_info['experiment_name']
    exist_ok = output_info['exist_ok']
    
    # force_timestamp is ignored (legacy param kept for backward compatibility)
    return create_experiment_dir(experiment_name, exist_ok, False)


def get_test_experiment_output_dir(config: Dict[str, Any]) -> Path:
    """
    Get test experiment output directory based on config

    Args:
        config: Configuration dictionary

    Returns:
        experiment_dir: Path to test experiment directory
    """
    output_info = get_output_info(config)
    experiment_name = output_info['experiment_name']
    exist_ok = output_info['exist_ok']

    fixed_subdir = config.get('_fixed_test_subdir_name')
    return create_test_experiment_dir(experiment_name, exist_ok, fixed_subdir)


def find_latest_train_experiment_dir(base_experiment_name: str) -> Optional[Path]:
    """
    Find the latest training experiment directory that starts with the given base name

    Args:
        base_experiment_name: Base experiment name (without timestamp)

    Returns:
        Path to latest matching directory under runs/train or None if not found
    """
    train_root = Path("runs") / "train"
    base_dir = train_root / base_experiment_name
    if not base_dir.exists():
        return None

    prefix = f"{base_experiment_name}_"
    candidates: List[Path] = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not candidates:
        return None

    # Choose by most recent modification time
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return latest


def create_config_template(output_path: str = "config_template.yaml") -> None:
    """
    Create a configuration template file
    
    Args:
        output_path: Path to save template file
    """
    template = {
        'dataset': {
            'train_txt': '/path/to/train.txt',
            'val_txt': '/path/to/val.txt',
            'test_txt': '/path/to/test.txt',
            'classes': {
                0: 'good',
                1: 'bad'
            },
            'num_classes': 2,
            'augmentation': {
                'transform_type': 'standard',
                'resize': [224, 224],
                'random_horizontal_flip': 0.5,
                'random_rotation': 10,
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2,
                    'hue': 0.1
                },
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }
        },
        'model': {
            'name': 'resnet18',
            'pretrained': True
        },
        'training': {
            'epochs': 50,
            'batch_size': 32,
            'learning_rate': 0.001,
            'num_workers': 4
        },
        'output': {
            'experiment_name': 'my_experiment',
            'exist_ok': False,
            'save_best_model': True,
            'save_training_history': True
        },
        'evaluation': {
            'save_confusion_matrix': True,
            'save_classification_report': True,
            'save_predictions': True
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(template, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"📄 Configuration template created: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration utilities')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--template', type=str, help='Create configuration template')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    
    args = parser.parse_args()
    
    if args.template:
        create_config_template(args.template)
    
    elif args.config:
        try:
            config = load_config(args.config)
            print("✅ Configuration loaded successfully")
            
            if args.validate:
                print("\nConfiguration Summary:")
                print(f"Dataset: {len(config['dataset']['classes'])} classes")
                print(f"Model: {config['model']['name']}")
                print(f"Training: {config['training']['epochs']} epochs")
                print(f"Output: {config['output']['experiment_name']}")
        
        except Exception as e:
            print(f"❌ Configuration error: {e}")
    
    else:
        print("Please provide --config or --template argument")
        print("\nUsage:")
        print("  python utils/config.py --template config.yaml")
        print("  python utils/config.py --config config.yaml --validate") 