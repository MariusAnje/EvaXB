import torch
from torch import nn
from modules import NModel
from modules import CrossLinear

class MLP3(NModel):
    def __init__(self):
        super().__init__()
        hidden = 8
        self.fc1 = CrossLinear(28*28, hidden)
        self.fc2 = CrossLinear(hidden,hidden)
        self.fc3 = CrossLinear(hidden,10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
