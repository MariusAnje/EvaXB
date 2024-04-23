import torch
from torch import nn
from modules import NModel
from modules import CrossLinear, CrossConv2d
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
        
        x = self.unpack_flattern(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        return x


class CIFAR(NModel):
    def __init__(self, N=6):
        super().__init__()
        N_weight=4
        N_ADC=4
        array_size=32

        self.conv1 = CrossConv2d(3, 64, 3, padding=1)
        self.conv2 = CrossConv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = CrossConv2d(64,128,3, padding=1)
        self.conv4 = CrossConv2d(128,128,3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = CrossConv2d(128,256,3, padding=1)
        self.conv6 = CrossConv2d(256,256,3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = CrossLinear(256 * 4 * 4, 1024)
        self.fc2 = CrossLinear(1024, 1024)
        self.fc3 = CrossLinear(1024, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv5(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        x = self.unpack_flattern(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x