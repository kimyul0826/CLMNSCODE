"""
Data augmentation and preprocessing transforms
"""

import torch
from torchvision import transforms
from torchvision.transforms import functional as TF
from typing import Dict, Any, List, Tuple


# ✅ 상단 crop 커스터마이즈 클래스
class TopCrop:
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, img):
        crop_height, crop_width = self.size
        return TF.crop(img, top=0, left=0, height=crop_height, width=crop_width)


# ✅ 하단 crop 커스터마이즈 클래스
class BottomCrop:
    def __init__(self, size):
        self.size = size  # (height, width)

    def __call__(self, img):
        crop_height, crop_width = self.size
        w, h = img.size
        # 하단에서 crop
        top = h - crop_height
        left = (w - crop_width) // 2  # 중앙 정렬
        return TF.crop(img, top=top, left=left, height=crop_height, width=crop_width)


# ✅ 커스텀 transform 정의 (비율 유지 + 패딩)
class ResizeWithPadding:
    def __init__(self, target_size, fill=0, padding_mode='constant'):
        self.target_size = target_size  # (height, width)
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size
        target_h, target_w = self.target_size

        # 비율 유지 resize
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = TF.resize(img, (new_h, new_w))

        # 패딩 계산
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        padding = (
            pad_w // 2, pad_h // 2,
            pad_w - pad_w // 2, pad_h - pad_h // 2
        )

        img = TF.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)
        return img


def get_train_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get training transforms based on configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_transforms: Training transforms
    """
    augmentation = config['dataset'].get('augmentation', {})

    # Kept for backward compatibility in direct calls.
    # Now the recommended path is to use build_transform_for_split(config, 'train').
    return build_transform_for_split(config, 'train')


def get_val_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get validation transforms based on configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        val_transforms: Validation transforms
    """
    # Kept for backward compatibility in direct calls.
    # Now the recommended path is to use build_transform_for_split(config, 'val').
    return build_transform_for_split(config, 'val')


def get_test_transforms(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get test transforms based on configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        test_transforms: Test transforms
    """
    # Test transforms are the same as validation transforms
    return get_val_transforms(config)


def get_transforms_by_split(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Get transforms for all splits
    
    Args:
        config: Configuration dictionary
    
    Returns:
        transforms_dict: Dictionary with transforms for each split
    """
    return {
        'train': get_train_transforms(config),
        'val': get_val_transforms(config),
        'test': get_test_transforms(config)
    }


# ✅ config 기반 transform 선택 함수들
def _get_resize_from_config(config: Dict[str, Any]) -> List[int]:
    """Get resize [H, W] from either dataset.resize or dataset.augmentation.resize."""
    dataset_cfg = config.get('dataset', {})
    if 'resize' in dataset_cfg and dataset_cfg['resize']:
        return dataset_cfg['resize']
    aug = dataset_cfg.get('augmentation', {})
    return aug.get('resize', [224, 224])


def get_crop_transform(config: Dict[str, Any], crop_type: str = 'center') -> transforms.Compose:
    """
    Get crop transform based on config and crop type
    
    Args:
        config: Configuration dictionary
        crop_type: 'center', 'top', 'bottom'
    
    Returns:
        transform: Crop transform
    """
    dataset_cfg = config.get('dataset', {})
    augmentation = dataset_cfg.get('augmentation', {})
    resize_size = _get_resize_from_config(config)
    normalize_mean = augmentation.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
    normalize_std = augmentation.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
    
    transform_list = [transforms.Resize(256)]
    
    if crop_type == 'center':
        transform_list.append(transforms.CenterCrop(resize_size))
    elif crop_type == 'top':
        transform_list.append(TopCrop(resize_size))
    elif crop_type == 'bottom':
        transform_list.append(BottomCrop(resize_size))
    else:
        raise ValueError(f"Unknown crop type: {crop_type}")
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])
    
    return transforms.Compose(transform_list)


def get_padding_transform(config: Dict[str, Any]) -> transforms.Compose:
    """
    Get padding transform based on config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        transform: Padding transform
    """
    dataset_cfg = config.get('dataset', {})
    augmentation = dataset_cfg.get('augmentation', {})
    resize_size = _get_resize_from_config(config)
    normalize_mean = augmentation.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
    normalize_std = augmentation.get('normalize', {}).get('std', [0.229, 0.224, 0.225])
    
    return transforms.Compose([
        ResizeWithPadding(resize_size, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std)
    ])


def get_transform_by_config(config: Dict[str, Any], split: str = 'train') -> transforms.Compose:
    """
    Get transform based on config settings
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', 'test'
    
    Returns:
        transform: Transform based on config
    """
    return build_transform_for_split(config, split)


def get_transforms_by_config(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Get transforms for all splits based on config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        transforms_dict: Dictionary with transforms for each split
    """
    return {
        'train': get_transform_by_config(config, 'train'),
        'val': get_transform_by_config(config, 'val'),
        'test': get_transform_by_config(config, 'test')
    }


# ✅ 새로운 함수: 각 split별로 다른 transform 지정
def get_split_specific_transforms(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Get split-specific transforms based on config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        transforms_dict: Dictionary with transforms for each split
    """
    augmentation = config['dataset'].get('augmentation', {})
    
    # 각 split별 transform 설정 확인
    train_transform_type = augmentation.get('train_transform', 'standard')
    val_transform_type = augmentation.get('val_transform', 'standard')
    test_transform_type = augmentation.get('test_transform', 'standard')
    
    transforms_dict = {}
    
    # Train transform
    if train_transform_type == 'standard':
        transforms_dict['train'] = get_train_transforms(config)
    elif train_transform_type in ['center_crop', 'top_crop', 'bottom_crop']:
        crop_type = train_transform_type.replace('_crop', '')
        transforms_dict['train'] = get_crop_transform(config, crop_type)
    elif train_transform_type == 'padding':
        transforms_dict['train'] = get_padding_transform(config)
    else:
        raise ValueError(f"Unknown train transform type: {train_transform_type}")
    
    # Val transform
    if val_transform_type == 'standard':
        transforms_dict['val'] = get_val_transforms(config)
    elif val_transform_type in ['center_crop', 'top_crop', 'bottom_crop']:
        crop_type = val_transform_type.replace('_crop', '')
        transforms_dict['val'] = get_crop_transform(config, crop_type)
    elif val_transform_type == 'padding':
        transforms_dict['val'] = get_padding_transform(config)
    else:
        raise ValueError(f"Unknown val transform type: {val_transform_type}")
    
    # Test transform
    if test_transform_type == 'standard':
        transforms_dict['test'] = get_val_transforms(config)  # Test uses val transforms
    elif test_transform_type in ['center_crop', 'top_crop', 'bottom_crop']:
        crop_type = test_transform_type.replace('_crop', '')
        transforms_dict['test'] = get_crop_transform(config, crop_type)
    elif test_transform_type == 'padding':
        transforms_dict['test'] = get_padding_transform(config)
    else:
        raise ValueError(f"Unknown test transform type: {test_transform_type}")
    
    return transforms_dict


def get_transforms_by_split_config(config: Dict[str, Any]) -> Dict[str, transforms.Compose]:
    """
    Get transforms based on split-specific config or fallback to general config
    
    Args:
        config: Configuration dictionary
    
    Returns:
        transforms_dict: Dictionary with transforms for each split
    """
    # Always build using the new ordered pipeline (base transform -> augmentation -> tensor/normalize)
    return {
        'train': build_transform_for_split(config, 'train'),
        'val': build_transform_for_split(config, 'val'),
        'test': build_transform_for_split(config, 'test'),
    }


# ===== New ordered pipeline helpers =====
def _canonicalize_transform_type(name: str) -> str:
    """Map user-facing names to internal canonical names.
    Accepted (case-insensitive): standard, center, top, bottom, padding,
    as well as legacy: center_crop, top_crop, bottom_crop.
    Returns one of: standard, center_crop, top_crop, bottom_crop, padding.
    """
    if not name:
        return 'standard'
    s = str(name).strip().lower()
    if s in ('standard',):
        return 'standard'
    if s in ('center', 'centre', 'center_crop', 'centre_crop'):
        return 'center_crop'
    if s in ('top', 'top_crop'):
        return 'top_crop'
    if s in ('bottom', 'bottom_crop', 'bot', 'down'):
        return 'bottom_crop'
    if s in ('padding', 'pad', 'padded'):
        return 'padding'
    # fallback to original string to trigger error upstream if unknown
    return s


def _get_transform_type_for_split(config: Dict[str, Any], split: str) -> str:
    dataset_cfg = config.get('dataset', {})
    # Prefer new location: dataset.transforms.{train|val|test}
    transforms_cfg = dataset_cfg.get('transforms', {})
    if isinstance(transforms_cfg, dict):
        val = transforms_cfg.get(split)
        if val:
            return _canonicalize_transform_type(val)
    # Backward compatibility: dataset.augmentation.{train_transform|val_transform|test_transform}
    augmentation = dataset_cfg.get('augmentation', {})
    key = f"{split}_transform"
    if key in augmentation and augmentation[key]:
        return _canonicalize_transform_type(augmentation[key])
    # Fallback default/base
    if 'transform_type' in augmentation:
        return _canonicalize_transform_type(augmentation.get('transform_type'))
    return 'standard'


def _build_yolov5_augmentation_transforms(config: Dict[str, Any]) -> List[Any]:
    dataset_cfg = config.get('dataset', {})
    augmentation = dataset_cfg.get('augmentation', {})

    # Default to no augmentation if not explicitly configured
    fliplr_p = augmentation.get('fliplr', augmentation.get('random_horizontal_flip', 0.0))
    flipud_p = augmentation.get('flipud', 0.0)

    degrees = augmentation.get('degrees', augmentation.get('random_rotation', 0))
    translate_frac = augmentation.get('translate', 0.0)
    scale_delta = augmentation.get('scale', 0.0)
    shear_deg = augmentation.get('shear', 0.0)
    perspective_scale = augmentation.get('perspective', 0.0)

    hsv_h = augmentation.get('hsv_h', augmentation.get('color_jitter', {}).get('hue', 0.0))
    hsv_s = augmentation.get('hsv_s', augmentation.get('color_jitter', {}).get('saturation', 0.0))
    hsv_v = augmentation.get('hsv_v', augmentation.get('color_jitter', {}).get('brightness', 0.0))

    aug_list: List[Any] = []

    if perspective_scale and perspective_scale > 0:
        aug_list.append(transforms.RandomPerspective(distortion_scale=float(perspective_scale), p=1.0))

    affine_translate = (translate_frac, translate_frac) if translate_frac and translate_frac > 0 else None
    affine_scale = (max(0.0, 1.0 - float(scale_delta)), 1.0 + float(scale_delta)) if scale_delta and scale_delta > 0 else None
    affine_shear = (-float(shear_deg), float(shear_deg)) if shear_deg and shear_deg != 0 else None

    if any(v is not None for v in (affine_translate, affine_scale, affine_shear)) or (degrees and degrees != 0):
        aug_list.append(
            transforms.RandomAffine(
                degrees=float(degrees) if degrees is not None else 0.0,
                translate=affine_translate,
                scale=affine_scale,
                shear=affine_shear,
            )
        )

    if fliplr_p and fliplr_p > 0:
        aug_list.append(transforms.RandomHorizontalFlip(p=float(fliplr_p)))
    if flipud_p and flipud_p > 0:
        aug_list.append(transforms.RandomVerticalFlip(p=float(flipud_p)))

    if any(x and x > 0 for x in (hsv_v, hsv_s, hsv_h)):
        aug_list.append(
            transforms.ColorJitter(
                brightness=float(hsv_v) if hsv_v is not None else 0.0,
                contrast=0.0,
                saturation=float(hsv_s) if hsv_s is not None else 0.0,
                hue=float(hsv_h) if hsv_h is not None else 0.0,
            )
        )

    return aug_list


def build_transform_for_split(config: Dict[str, Any], split: str = 'train') -> transforms.Compose:
    dataset_cfg = config.get('dataset', {})
    augmentation = dataset_cfg.get('augmentation', {})
    resize_hw = _get_resize_from_config(config)
    normalize_mean = augmentation.get('normalize', {}).get('mean', [0.485, 0.456, 0.406])
    normalize_std = augmentation.get('normalize', {}).get('std', [0.229, 0.224, 0.225])

    transform_type = _get_transform_type_for_split(config, split)

    pipeline: List[Any] = []

    # 1) Base transform: geometry/region specification
    if transform_type == 'standard':
        pipeline.append(transforms.Resize(resize_hw))
    elif transform_type in ['center_crop', 'top_crop', 'bottom_crop']:
        # Follow the earlier convention: Resize(256) then crop to target size
        pipeline.append(transforms.Resize(256))
        crop_size = _get_resize_from_config(config)
        if transform_type == 'center_crop':
            pipeline.append(transforms.CenterCrop(crop_size))
        elif transform_type == 'top_crop':
            pipeline.append(TopCrop(crop_size))
        elif transform_type == 'bottom_crop':
            pipeline.append(BottomCrop(crop_size))
    elif transform_type == 'padding':
        pipeline.append(ResizeWithPadding(resize_hw, fill=0, padding_mode='constant'))
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")

    # 2) Augmentations (train only)
    if split == 'train':
        pipeline.extend(_build_yolov5_augmentation_transforms(config))

    # 3) Tensor conversion and normalization
    pipeline.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=normalize_mean, std=normalize_std),
    ])

    return transforms.Compose(pipeline)


# Removed: all preset helpers and CLI presets per requirement


if __name__ == "__main__":
    # Example usage
    import argparse
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description='Transform utilities')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    args = parser.parse_args()
    
    if args.config:
        try:
            from utils.config import load_config
            config = load_config(args.config)
            transforms_dict = get_transforms_by_config(config)
            print("Transforms loaded from config:")
            for split, transform in transforms_dict.items():
                print(f"  {split}: {transform}")
        except ImportError:
            print("Error: Could not import config module. Make sure you're running from the correct directory.")
            print("Try running: python -m utils.transforms --config config.yaml")
        except Exception as e:
            print(f"Error loading config: {e}")
    
    
    
    else:
        print("Please provide --config argument")
        print("\nConfig transform types:")
        print("  - standard: Standard (resize)")
        print("  - center: Center crop")
        print("  - top: Top crop")
        print("  - bottom: Bottom crop")
        print("  - padding: Resize with padding")