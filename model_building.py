import torch
import torch.nn as nn
from torchvision.models import resnet50

# Model Initialization
base_model = resnet50(weights='IMAGENET1K_V1')

# Freezing Layers
for param in base_model.parameters():
    param.requires_grad = False

# Building Model and adjust the required parameters
model = nn.Sequential(
    base_model,
    nn.Dropout(0.5),
    nn.Flatten(),
    nn.BatchNorm1d(2048),
    nn.Linear(2048, 2048),
    nn.BatchNorm1d(2048),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(2048, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 4)
)
