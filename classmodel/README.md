# ClassModel - 분류 모델 학습 및 평가 프레임워크

## 🎯 **4가지 실행 방식**

### 1. **Train만 실행**
```bash
python main.py --config config.yaml --mode train --name my_experiment
```
**네이밍**: `my_experiment` → 중복 시 `my_experiment_1`, `my_experiment_2`

### 2. **Test만 실행**
```bash
python main.py --config config.yaml --mode evaluate --name my_experiment --model_path runs/train/my_experiment/models/best_model.pth
```
**네이밍**: `my_experiment` → 중복 시 `my_experiment_1`, `my_experiment_2`

### 3. **Train과 Test 동시 실행**
```bash
python main.py --config config.yaml --mode train_evaluate --name my_experiment
```
**네이밍**: 
- Train: `my_experiment` → 중복 시 `my_experiment_1`
- Test: Train의 실제 실험명을 자동으로 따라감

 

## 📁 **폴더 구조 예시**

```
runs/
├── train/
│   ├── my_experiment/           # 1번째 실행
│   │   ├── models/
│   │   │   ├── best_model.pth
│   │   │   └── final_model.pth
│   │   ├── plots/
│   │   └── logs/
│   ├── my_experiment_1/         # 2번째 실행 (중복 시)
│   │   └── ...
│   └── ...
│   └── ...
├── test/
│   ├── my_experiment/           # Test 결과 (Train과 동일한 실험명)
│   │   ├── predictions/
│   │   ├── plots/
│   │   └── results/
│   ├── my_experiment_1/         # Test 결과 (Train과 동일한 실험명)
│   │   └── ...
│   └── ...
│   └── ...
└── ...
```

## 🚀 **사용법**

### 기본 설정 파일
```yaml
# config.yaml
dataset:
  train_txt: "/path/to/train.txt"
  val_txt: "/path/to/val.txt"
  test_txt: "/path/to/test.txt"
  classes:
    0: "good"
    1: "bad"
  augmentation:
    transform_type: "standard"
    resize: [224, 224]

model:
  name: "resnet18"
  pretrained: true

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

output:
  experiment_name: "my_experiment"  # --name으로 오버라이드 가능
  exist_ok: false
```

### 1. Train만 실행
```bash
python main.py --config config.yaml --mode train --name my_experiment
```

### 2. Test만 실행
```bash
python main.py --config config.yaml --mode evaluate --name my_experiment --model_path runs/train/my_experiment/models/best_model.pth
```

### 3. Train과 Test 동시 실행
```bash
python main.py --config config.yaml --mode train_evaluate --name my_experiment
```

 

## 📋 **네이밍 규칙**

| 실행 방식 | Train 네이밍 | Test 네이밍 | 중복 처리 |
|-----------|-------------|------------|-----------|
| **Train만** | `my_experiment` | - | `my_experiment_1`, `my_experiment_2` |
| **Test만** | - | `my_experiment` | `my_experiment_1`, `my_experiment_2` |
| **Train+Test** | `my_experiment` | Train의 실제 실험명 | `my_experiment_1`, `my_experiment_2` |
 

## 🎯 **특징**

1. **일관성**: Train과 Test가 동일한 실험명 사용
2. **중복 방지**: 일반 실행은 증가하는 숫자 방식으로 중복 방지
3. **명확한 구분**: `runs/train/`과 `runs/test/`로 명확히 분리
4. **유연성**: `--name` 옵션으로 실험명 오버라이드 가능

## 구조

```
classifiers/
├── main.py                    # 모델 선택 + 전체 실행 제어
├── config.yaml                # 설정 파일 (yaml)
├── models/
│   ├── __init__.py
│   ├── resnet.py              # ResNet18, ResNet50
│   ├── efficientnet.py        # EfficientNet
│   └── mobilenet.py           # MobileNet
├── train.py                   # 학습 루프
├── evaluate.py                # 평가 + confusion matrix 등
├── utils/
│   ├── __init__.py
│   ├── config.py              # 설정 파일 관리
│   ├── dataset.py             # Dataset 불러오기/전처리 포함
│   ├── transforms.py          # 전처리 정의
│   └── plot.py                # 시각화 코드
├── requirements.txt           # 필요한 패키지들
├── runs/                      # 실험 결과 저장 폴더
│   ├── my_experiment/         # 실험별 폴더
│   │   ├── models/            # 학습된 모델들
│   │   ├── plots/             # 그래프들
│   │   ├── results/           # 평가 결과
│   │   ├── logs/              # 로그 파일들
│   │   └── experiment_summary.txt  # 실험 요약
│   └── another_experiment/    # 다른 실험
└── README.md                 # 이 파일
```

## 설치

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

## 설정 파일 준비

프로젝트는 yaml 설정 파일을 사용합니다. 설정 파일 템플릿을 생성하려면:

```bash
python utils/config.py --template config.yaml
```

또는 직접 `config.yaml` 파일을 생성하세요:

```yaml
# Dataset Configuration
dataset:
  # Dataset paths
  train_txt: "/path/to/train.txt"
  val_txt: "/path/to/val.txt"
  test_txt: "/path/to/test.txt"
  
  # Class information
  classes:
    0: "good"      # Class 0: Good/Normal samples
    1: "bad"       # Class 1: Bad/Defective samples
  
  # Number of classes (automatically calculated from classes dict)
  num_classes: 2
  
  # Data augmentation settings
  augmentation:
    # Transform type selection
    # Options: standard, center_crop, top_crop, bottom_crop, padding, ensemble
    transform_type: "standard"  # Choose your transform method
    
    # Standard augmentation settings (used when transform_type is "standard")
    resize: [224, 224]
    random_horizontal_flip: 0.5
    random_rotation: 10
    color_jitter:
      brightness: 0.2
      contrast: 0.2
      saturation: 0.2
      hue: 0.1
    normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

# Model Configuration
model:
  name: "resnet18"  # Options: resnet18, resnet50, efficientnet, mobilenet
  pretrained: true
  
# Training Configuration
training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001
  num_workers: 4
  
# Output Configuration
output:
  experiment_name: "my_experiment"  # Name for the experiment (will create runs/my_experiment/)
  exist_ok: false                   # Whether to overwrite existing experiment directory
  save_best_model: true
  save_training_history: true
```

## 데이터셋 준비

데이터셋은 train.txt, val.txt, test.txt 파일에 이미지 경로와 클래스 인덱스가 작성된 형태로 준비해야 합니다:

```
/path/to/image1.jpg 0
/path/to/image2.jpg 1
/path/to/image3.jpg 0
/path/to/image4.jpg 1
...
```

## 사용법

### 1. 모델 학습

```bash
python main.py --config config.yaml --mode train
```

### 2. 모델 평가

```bash
python main.py --config config.yaml --mode evaluate --model_path runs/my_experiment/models/best_model.pth
```

### 3. 학습 + 평가

```bash
python main.py --config config.yaml --mode train_evaluate
```

### 4. 설정 오버라이드

```bash
python main.py --config config.yaml --override --epochs 100 --batch_size 16 --lr 0.0001
```



## 주요 옵션

- `--config`: 설정 파일 경로 (필수)
- `--mode`: 실행 모드 (train, evaluate, train_evaluate)
- `--model_path`: 평가할 모델 경로
- `--override`: 명령행 인수로 설정 오버라이드
- `--epochs`: 학습 에포크 수 오버라이드
- `--batch_size`: 배치 크기 오버라이드
- `--lr`: 학습률 오버라이드

## 지원하는 모델

1. **ResNet18**: 빠르고 효율적인 기본 모델
2. **ResNet50**: 더 깊은 네트워크로 높은 정확도
3. **EfficientNet**: 효율적인 아키텍처
4. **MobileNet**: 경량화된 모델

## 전처리 (Transforms)

프레임워크는 5가지 기본 전처리 방식을 지원합니다. `config.yaml`의 `augmentation.transform_type`에서 선택할 수 있습니다:

### Transform Type 옵션들

#### 1. **standard** (기본값)
- **설명**: 표준 데이터 증강 (Resize + RandomHorizontalFlip + RandomRotation + ColorJitter)
- **사용**: 일반적인 이미지 분류 작업
- **설정**:
```yaml
augmentation:
  transform_type: "standard"
  resize: [224, 224]
  random_horizontal_flip: 0.5
  random_rotation: 10
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
```

#### 2. **center_crop**
- **설명**: Resize(256) + CenterCrop(224) - 이미지 중앙 부분 추출
- **사용**: 중앙에 중요한 정보가 있는 이미지 (예: 프론트펜더)
- **설정**:
```yaml
augmentation:
  transform_type: "center_crop"
  resize: [224, 224]
```

#### 3. **top_crop**
- **설명**: Resize(256) + TopCrop(224) - 이미지 상단 부분 추출
- **사용**: 상단에 중요한 정보가 있는 이미지 (예: 후드)
- **설정**:
```yaml
augmentation:
  transform_type: "top_crop"
  resize: [224, 224]
```

#### 4. **bottom_crop**
- **설명**: Resize(256) + BottomCrop(224) - 이미지 하단 부분 추출
- **사용**: 하단에 중요한 정보가 있는 이미지 (예: 트렁크)
- **설정**:
```yaml
augmentation:
  transform_type: "bottom_crop"
  resize: [224, 224]
```

#### 5. **padding**
- **설명**: ResizeWithPadding(224x224) - 비율 유지하며 패딩으로 크기 맞춤
- **사용**: 이미지 비율을 유지해야 하는 경우
- **설정**:
```yaml
augmentation:
  transform_type: "padding"
  resize: [224, 224]
```

### Split-Specific Transforms (각 split별 다른 transform)

각 split(train, val, test)에 대해 서로 다른 transform을 지정할 수 있습니다:

```yaml
augmentation:
  # Split-specific transforms (각 split별 다른 transform)
  train_transform: "standard"      # Train: 표준 증강
  val_transform: "padding"         # Val: 패딩
  test_transform: "center_crop"    # Test: 중앙 crop
```

#### Split-Specific Transform 옵션들

| Transform | 설명 | 사용 예시 |
|-----------|------|-----------|
| **standard** | 표준 증강 (train) / 무증강 (val/test) | 일반적인 학습 |
| **center_crop** | 중앙 부분 추출 | 프론트펜더 |
| **top_crop** | 상단 부분 추출 | 후드 |
| **bottom_crop** | 하단 부분 추출 | 트렁크 |
| **padding** | 비율 유지 + 패딩 | 비율 중요 |

#### Split-Specific Transform 사용 예시

```yaml
# 예시 1: 각 split별 다른 crop 방식
augmentation:
  train_transform: "top_crop"      # Train: 상단 crop
  val_transform: "center_crop"     # Val: 중앙 crop
  test_transform: "bottom_crop"    # Test: 하단 crop

# 예시 2: 혼합 transform
augmentation:
  train_transform: "standard"         # Train: 표준 증강
  val_transform: "padding"            # Val: 패딩
  test_transform: "center_crop"       # Test: 중앙 crop
```

### 커스텀 전처리 클래스
- **TopCrop**: 이미지 상단에서 crop
- **BottomCrop**: 이미지 하단에서 crop  
- **ResizeWithPadding**: 비율 유지하며 패딩으로 크기 맞춤

### 전처리 사용 예시

```python
from utils.transforms import get_door_ensemble_transforms

# 도어 앙상블 전처리 가져오기
transforms_dict = get_door_ensemble_transforms()

# 각 전처리 방식 사용
high_transform = transforms_dict['high']
mid_transform = transforms_dict['mid']
low_transform = transforms_dict['low']
bottom_transform = transforms_dict['bottom']
```

### 전처리 설정 확인

```bash
# 설정 파일 기반 전처리 확인
python utils/transforms.py --config config.yaml

# 프리셋 전처리 확인
python utils/transforms.py --preset door_ensemble

# 또는 모듈로 실행 (import 오류 시)
python -m utils.transforms --config config.yaml
```

## 출력 파일 구조

모든 실험 결과는 `runs/` 폴더에 실험명으로 저장됩니다:

```
runs/
├── my_experiment/              # 실험명 폴더
│   ├── models/                 # 학습된 모델들
│   │   ├── best_model.pth      # 최고 성능 모델
│   │   └── final_model.pth     # 최종 모델
│   ├── plots/                  # 그래프들
│   │   ├── training_history.png    # 학습 히스토리
│   │   ├── learning_curves.png     # 학습 곡선
│   │   ├── confusion_matrix.png    # 혼동 행렬
│   │   └── per_class_accuracy.png # 클래스별 정확도
│   ├── results/                # 평가 결과
│   │   ├── evaluation_results.json     # 평가 결과 JSON
│   │   └── classification_report.txt   # 분류 리포트
│   ├── logs/                   # 로그 파일들
│   │   └── training_history.json      # 학습 히스토리 JSON
│   ├── experiment_summary.txt   # 실험 요약
│   └── evaluation_summary.txt   # 평가 요약
└── another_experiment/         # 다른 실험
    └── ...
```

## 예제

### 프론트펜더 데이터로 ResNet18 학습 (CenterCrop)

```yaml
# config.yaml
dataset:
  train_txt: "/path/to/frontfender/train.txt"
  val_txt: "/path/to/frontfender/val.txt"
  test_txt: "/path/to/frontfender/test.txt"
  classes:
    0: "good"
    1: "bad"
  augmentation:
    transform_type: "center_crop"  # 중앙 부분 추출
    resize: [224, 224]

model:
  name: "resnet18"
  pretrained: true

training:
  epochs: 100
  batch_size: 16
  learning_rate: 0.0001

output:
  experiment_name: "frontfender_resnet18"  # 실험명
  exist_ok: false
```

### 후드 데이터로 EfficientNet 학습 (TopCrop)

```yaml
# config.yaml
dataset:
  train_txt: "/path/to/hood/train.txt"
  val_txt: "/path/to/hood/val.txt"
  test_txt: "/path/to/hood/test.txt"
  classes:
    0: "good"
    1: "bad"
  augmentation:
    transform_type: "top_crop"  # 상단 부분 추출
    resize: [224, 224]

model:
  name: "efficientnet"
  pretrained: true

training:
  epochs: 80
  batch_size: 24
  learning_rate: 0.001

output:
  experiment_name: "hood_efficientnet"  # 실험명
  exist_ok: false
```

### 트렁크 데이터로 MobileNet 학습 (BottomCrop)

```yaml
# config.yaml
dataset:
  train_txt: "/path/to/trunk/train.txt"
  val_txt: "/path/to/trunk/val.txt"
  test_txt: "/path/to/trunk/test.txt"
  classes:
    0: "good"
    1: "bad"
  augmentation:
    transform_type: "bottom_crop"  # 하단 부분 추출
    resize: [224, 224]

model:
  name: "mobilenet"
  pretrained: true

training:
  epochs: 60
  batch_size: 32
  learning_rate: 0.001

output:
  experiment_name: "trunk_mobilenet"  # 실험명
  exist_ok: false
```

### Split-Specific Transform 실험

```yaml
# config.yaml - 각 split별 다른 transform
dataset:
  train_txt: "/path/to/split_specific/train.txt"
  val_txt: "/path/to/split_specific/val.txt"
  test_txt: "/path/to/split_specific/test.txt"
  classes:
    0: "good"
    1: "bad"
  augmentation:
    # 각 split별 다른 transform 지정
    train_transform: "standard"      # Train: 표준 증강
    val_transform: "padding"         # Val: 패딩
    test_transform: "center_crop"    # Test: 중앙 crop
    resize: [224, 224]

model:
  name: "resnet18"
  pretrained: true

training:
  epochs: 50
  batch_size: 32
  learning_rate: 0.001

output:
  experiment_name: "split_specific_experiment"  # 실험명
  exist_ok: false
```

### 혼합 Transform 실험

```yaml
# config.yaml - 혼합 transform
dataset:
  train_txt: "/path/to/mixed/train.txt"
  val_txt: "/path/to/mixed/val.txt"
  test_txt: "/path/to/mixed/test.txt"
  classes:
    0: "good"
    1: "bad"
  augmentation:
    # 혼합 transform 조합
    train_transform: "standard"         # Train: 표준 증강
    val_transform: "padding"            # Val: 패딩
    test_transform: "center_crop"       # Test: 중앙 crop
    resize: [224, 224]

model:
  name: "efficientnet"
  pretrained: true

training:
  epochs: 80
  batch_size: 24
  learning_rate: 0.001

output:
  experiment_name: "mixed_transform_experiment"  # 실험명
  exist_ok: false
```



## 데이터셋 정보 확인

```bash
python utils/dataset.py --config config.yaml
```

## 데이터셋 검증

```bash
python utils/dataset.py --config config.yaml --validate
```

## 설정 파일 검증

```bash
python utils/config.py --config config.yaml
```

## 전처리 설정 확인

```bash
python utils/transforms.py --config config.yaml
```

## 평가 결과 확인

```bash
python evaluate.py --config config.yaml --model_path runs/my_experiment/models/best_model.pth
```

## 시각화

```bash
python utils/plot.py \
    --history_file runs/my_experiment/logs/training_history.json \
    --results_file runs/my_experiment/results/evaluation_results.json \
    --output_dir runs/my_experiment/plots
```

## 주의사항

1. GPU 사용을 권장합니다 (CUDA 지원)
2. txt 파일의 이미지 경로가 올바른지 확인하세요
3. 이미지 파일은 jpg, jpeg, png, bmp, tiff 형식을 지원합니다
4. 메모리 부족 시 배치 크기를 줄이세요
5. txt 파일의 인코딩은 UTF-8을 사용하세요
6. train.txt, val.txt, test.txt 파일이 모두 필요합니다
7. 클래스 인덱스는 0부터 시작하는 정수여야 합니다
8. 설정 파일의 경로가 올바른지 확인하세요
9. transform_type은 다음 중 하나를 선택하세요: standard, center_crop, top_crop, bottom_crop, padding, ensemble
10. import 오류 발생 시 `python -m utils.xxx` 형태로 실행하세요
11. 실험 결과는 `runs/` 폴더에 실험명으로 저장됩니다

## 문제 해결

### CUDA 메모리 부족
```bash
python main.py --config config.yaml --override --batch_size 8
```

### 학습이 느린 경우
```bash
python main.py --config config.yaml --override --batch_size 64
```

### 과적합 방지
```bash
python main.py --config config.yaml --override --epochs 30 --lr 0.0001
```

### 이미지 경로 오류
```bash
# 데이터셋 검증으로 누락된 이미지 확인
python utils/dataset.py --config config.yaml --validate

# 또는 모듈로 실행
python -m utils.dataset --config config.yaml --validate
```

### 설정 파일 오류
```bash
# 설정 파일 검증
python utils/config.py --config config.yaml
```

### Transform 타입 오류
```bash
# 지원되는 transform 타입 확인
python utils/transforms.py --config config.yaml

# 또는 모듈로 실행
python -m utils.transforms --config config.yaml
```

### Import 오류
```bash
# 상대 import 오류 발생 시 모듈로 실행
python -m utils.transforms --config config.yaml
python -m utils.dataset --config config.yaml --validate
python -m utils.config --config config.yaml
```

### 실험 결과 확인
```bash
# 실험 폴더 구조 확인
ls -la runs/my_experiment/

# 모델 파일 확인
ls -la runs/my_experiment/models/

# 그래프 확인
ls -la runs/my_experiment/plots/

# 결과 확인
ls -la runs/my_experiment/results/
```

## txt 파일 생성 예시

Python으로 txt 파일을 생성하는 예시:

```python
import os

# 데이터 분할 함수
def split_data(image_paths, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """데이터를 train, val, test로 분할"""
    from sklearn.model_selection import train_test_split
    
    # First split: separate test set
    train_val_paths, test_paths = train_test_split(
        image_paths, test_size=test_ratio, random_state=42
    )
    
    # Second split: separate validation set
    train_paths, val_paths = train_test_split(
        train_val_paths, test_size=val_ratio/(1-test_ratio), random_state=42
    )
    
    return train_paths, val_paths, test_paths

# 클래스별로 이미지 경로 수집 (클래스 인덱스 사용)
class_images = {
    0: ['/path/to/good1.jpg', '/path/to/good2.jpg', '/path/to/good3.jpg'],  # Class 0
    1: ['/path/to/bad1.jpg', '/path/to/bad2.jpg', '/path/to/bad3.jpg']      # Class 1
}

# 각 클래스별로 데이터 분할
splits = {}
for class_index, image_paths in class_images.items():
    train_paths, val_paths, test_paths = split_data(image_paths)
    splits[class_index] = {
        'train': train_paths,
        'val': val_paths,
        'test': test_paths
    }

# txt 파일 생성
for split_name in ['train', 'val', 'test']:
    with open(f'{split_name}.txt', 'w', encoding='utf-8') as f:
        for class_index, class_paths in splits.items():
            for path in class_paths[split_name]:
                f.write(f'{path} {class_index}\n')

print("txt files created: train.txt, val.txt, test.txt")
```

## 클래스 인덱스 매핑

- `Class_0`: 첫 번째 클래스 (예: good, 정상)
- `Class_1`: 두 번째 클래스 (예: bad, 불량)
- `Class_2`: 세 번째 클래스 (3개 이상 클래스인 경우)

클래스 인덱스는 0부터 시작하며, 연속된 정수여야 합니다. 