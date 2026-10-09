"""
Simple CNN architectures for PathMNIST classification.
Includes MLP baseline and CNN variants for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    """
    Simple linear model.
    """
    
    def __init__(self, num_classes=9):
        super().__init__()
        # PathMNIST images are 3x28x28 = 2352 features
        input_size = 3 * 28 * 28
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        x_flatten = x.view(x.size(0), -1)
        output = self.linear(x_flatten)
        return output
        
class MLPModel(nn.Module):
    """
    Simple MLP model: Flatten input then run through hidden layers.
    """
    
    def __init__(self, num_classes=9):
        super(MLPModel, self).__init__()
        # PathMNIST images are 3x28x28 = 2352 features
        input_size = 3 * 28 * 28
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)   
        self.fc4 = nn.Linear(32, num_classes)     
        # TODO: Add your own MLP architecture here
    
    def forward(self, x):
        x_flatten = x.view(x.size(0), -1)
        out1 = self.fc1(x_flatten)
        out1 = F.relu(out1)
        out2 = self.fc2(out1)
        out2 = F.relu(out2)
        out3 = self.fc3(out2)
        out3 = F.relu(out3)
        output = self.fc4(out3)
        return output


class CNNModel1(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """
    
    def __init__(self, num_classes=9):
        super(CNNModel1, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(3, 32, 3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
            )
        self.linear = nn.Linear(64*7*7, num_classes)
        
        # TODO: Add your own CNN architecture here
    
    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x

class CNNModel2(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """
    
    def __init__(self, num_classes=9):
        super(CNNModel2, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(3, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
            )
        self.linear = nn.Linear(64*7*7, num_classes)
        
        # TODO: Add your own CNN architecture here
    
    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


class CNNModel3(nn.Module):
    """
    Simple CNN model: TODO: Add your own architecture here
    """
    
    def __init__(self, num_classes=9):
        super(CNNModel3, self).__init__()
        self.seq = nn.Sequential(nn.Conv2d(3, 32, 3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
            )
        self.linear = nn.Linear(128*3*3, num_classes)
        
        # TODO: Add your own CNN architecture here
    
    def forward(self, x):
        x = self.seq(x)
        x = self.linear(x)
        return x


    

def get_model(model_name, num_classes=9):
    """Get model by name."""
    if model_name == 'mlp':
        return MLPModel(num_classes)
    elif model_name == 'cnn1':
        return CNNModel1(num_classes)
    elif model_name == 'cnn2':
        return CNNModel2(num_classes)
    elif model_name == 'cnn3':
        return CNNModel3(num_classes)
    elif model_name == 'linear':
        return LinearModel(num_classes)
    else:
        #TODO: add your models names here
        raise ValueError("Unknown model: {}".format(model_name))

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
