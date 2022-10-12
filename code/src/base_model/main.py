import torch as t; 
import torch.nn as nn;
from parts import bounding_box, classes, encoded, confidence

# Inspired by: https://pytorch.org/tutorials/beginner/nn_tutorial.html#nn-sequential 
class BasisModel(nn.Module): 
 
    def forward(self, x): 
        enc = encoded(x)
        return t.cat((confidence(enc), bounding_box(enc), classes(enc)))

model = BasisModel(); 

#format: img[channel][row][column]
dims = 64 
img = t.rand((1,dims,dims)); 

print(f"{model.forward(img).shape=}");
