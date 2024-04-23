import torch
from torch import nn
from modules import NModel
from modules import CrossLinear, CrossConv2d

def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class MLP3(NModel):
    def __init__(self):
        super().__init__()
        hidden = 8
        N_weight=4
        N_ADC=4
        array_size=32
        self.fc1 = CrossLinear(28*28, hidden, N_weight=N_weight, N_ADC=N_ADC)
        self.fc2 = CrossLinear(hidden,hidden, N_weight=N_weight, N_ADC=N_ADC)
        self.fc3 = CrossLinear(hidden,10, N_weight=N_weight, N_ADC=N_ADC)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class LeNet(NModel):

    def __init__(self):
        super().__init__()
        N_weight=4
        N_ADC=4
        array_size=32
        self.conv1 = CrossConv2d(1, 6, 3, padding=1)
        self.conv2 = CrossConv2d(6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = CrossLinear(16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = CrossLinear(120, 84)
        self.fc3 = CrossLinear(84, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(-1, num_flat_features(x))
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x
