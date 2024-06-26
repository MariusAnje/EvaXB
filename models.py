import torch
from torch import nn
from modules import NModel
from modules import CrossLinear, CrossConv2d
class MLP3(NModel):
    def __init__(self, device_type="RRAM1"):
        super().__init__("MLP3", device_type)
        hidden = 8
        # self.N_weight=4
        # self.N_ADC=4
        # self.array_size=32
        self.fc1 = self.get_linear(28*28, hidden)
        self.fc2 = self.get_linear(hidden,hidden)
        self.fc3 = self.get_linear(hidden,10)
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

    def __init__(self, device_type="RRAM1"):
        super().__init__("LeNet", device_type)
        self.conv1 = self.get_conv2d(1, 6, 3, padding=1)
        self.conv2 = self.get_conv2d(6, 16, 3, padding=1)
        # an affine operation: y = Wx + b
        self.fc1 = self.get_linear(16 * 7 * 7, 120)  # 6*6 from image dimension
        self.fc2 = self.get_linear(120, 84)
        self.fc3 = self.get_linear(84, 10)
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
    def __init__(self, device_type="RRAM1"):
        super().__init__("CIFAR", device_type)
        # self.N_weight=6
        # self.N_ADC=6
        # self.array_size=64

        self.conv1 = self.get_conv2d(3, 64, 3, padding=1)
        self.conv2 = self.get_conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)

        self.conv3 = self.get_conv2d(64,128,3, padding=1)
        self.conv4 = self.get_conv2d(128,128,3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        self.conv5 = self.get_conv2d(128,256,3, padding=1)
        self.conv6 = self.get_conv2d(256,256,3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.fc1 = self.get_linear(256 * 4 * 4, 1024)
        self.fc2 = self.get_linear(1024, 1024)
        self.fc3 = self.get_linear(1024, 10)
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