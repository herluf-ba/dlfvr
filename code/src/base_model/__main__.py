import torch as t
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torch.nn.functional as F
from training import fit
from torchvision.transforms import ToTensor

from __init__ import BaseModel

bs = 64
lr = 0.001
momentum = 0.9
epochs = 10
loss_func = F.cross_entropy

mnist_train_ds = MNIST(
    root="../../.datasets",
    train=True,
    download=True,
)

mnist_test_ds = MNIST(root="../../.datasets",
                      train=False,
                      download=True,
                      transform=ToTensor())


def mod_mnist_dataset(ds):
    # Neg values = no image in quadrant
    mod_list = np.random.randint(-10, 9, size=(ds.__len__(), 2, 2))
    y = []


modded_mnist_dataset = mod_mnist_dataset(mnist_train_ds)

# TODO: Add transformation that creates bounding boxes and all that jazz for each image
train_dl = DataLoader(mnist_train_ds, batch_size=bs)  # 60.000 images
test_dl = DataLoader(mnist_test_ds)  # 10.000 images

model = BaseModel()
opt = t.optim.SGD(model.parameters(), lr=lr,
                  momentum=momentum)  # Stochastic Gradient Descent

fit(model, epochs, loss_func, opt, train_dl, test_dl)

print(f"{model.forward(img).shape=}")
