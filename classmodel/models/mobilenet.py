"""
MobileNet model for classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MobileNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_size='v2'):
        super(MobileNet, self).__init__()
        
        # Choose MobileNet variant
        if model_size == 'v2':
            self.model = models.mobilenet_v2(pretrained=pretrained)
        elif model_size == 'v3_small':
            self.model = models.mobilenet_v3_small(pretrained=pretrained)
        elif model_size == 'v3_large':
            self.model = models.mobilenet_v3_large(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported MobileNet size: {model_size}")
        
        # Modify the final layer for our number of classes
        if model_size == 'v2':
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
        else:  # v3 models
            num_features = self.model.classifier[3].in_features
            self.model.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.2),
                nn.Linear(num_features, num_classes)
            )
    
    def forward(self, x):
        return self.model(x) 