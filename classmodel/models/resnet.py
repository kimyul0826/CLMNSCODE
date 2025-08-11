"""
ResNet models for classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(pretrained=pretrained)
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet50, self).__init__()
        self.model = models.resnet50(pretrained=pretrained)
        
        # Modify the final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.model(x) 