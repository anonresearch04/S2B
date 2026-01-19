import torch
import torch.nn as nn
import numpy as np

class SketchModel(nn.Module):
    def __init__(self, max_length, label_size):
        super(SketchModel, self).__init__()
        torch.manual_seed(2025)
      
        self.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=label_size, 
            kernel_size=(max_length, 3001),                  
            stride=1, 
            padding=0
        )
        self.dropout = nn.Dropout(p=0.3)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        return x


    def get_sketch(self):
        sketch = self.conv1.weight
        sketch = sketch.detach().cpu().numpy()
        sketch = np.squeeze(sketch, axis=1)
        sketch = np.transpose(sketch, (1, 2, 0))
        sketch = np.ascontiguousarray(sketch)
        return sketch
    