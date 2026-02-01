import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__() #initialize nn.Module
        self.fc1 = nn.Linear(input_size, hidden_size)#Input layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)#hidden layer 1
        self.out = nn.Linear(hidden_size, output_size)#layer 2

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
    
class Cnn():
    def __init__(self, hidden_size1, hidden_size2, num_classes, seed=0):
        rng = np.random.default_rng(seed)
        # Conv1: (16, 1, 3, 3)
        self.W1 = rng.standard_normal((hidden_size1, 1, 3, 3)).astype(np.float32) * 0.1
        self.b1 = np.zeros((16,), dtype=np.float32)

        # Conv2: (32, 16, 3, 3)
        self.W2 = rng.standard_normal((hidden_size2, hidden_size1, 3, 3)).astype(np.float32) * 0.1
        self.b2 = np.zeros((32,), dtype=np.float32)

        # Linear after GAP: (32 -> num_classes)
        self.W3 = rng.standard_normal((hidden_size2, num_classes)).astype(np.float32) * 0.1
        self.b3 = np.zeros((num_classes,), dtype=np.float32)
        pass

    def print_weights(self):
        print("Conv1 weights shape:", self.W1.shape)
        print("Conv2 weights shape:", self.W2.shape)
        print("Linear weights shape:", self.W3.shape)
a = Cnn(16,32,10,0)
a.print_weights()