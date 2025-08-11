"""
EfficientNet model for classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EfficientNet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True, model_size='b0'):
        super(EfficientNet, self).__init__()
        
        # Choose EfficientNet variant
        if model_size == 'b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
        elif model_size == 'b1':
            self.model = models.efficientnet_b1(pretrained=pretrained)
        elif model_size == 'b2':
            self.model = models.efficientnet_b2(pretrained=pretrained)
        elif model_size == 'b3':
            self.model = models.efficientnet_b3(pretrained=pretrained)
        elif model_size == 'b4':
            self.model = models.efficientnet_b4(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported EfficientNet size: {model_size}")
        
        # Modify the final layer for our number of classes
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x) 