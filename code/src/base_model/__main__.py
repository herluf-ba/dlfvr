import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms import Compose, Grayscale, Resize
import numpy as np

from svhn import SVHN
from training import fit
from __init__ import BaseModel

batch_size = 64
learning_rate = 0.001
momentum = 0.9
epochs = 10
loss_func = F.mse_loss

S = 2
# Transform input images
transform = Compose([
    Grayscale(),
    # TODO: This resize breaks our labels. The coordinates are in the original image space.
    Resize((32 * S, 32 * S)),
])


def into_resized(original, p):
    return (p[0] / original[0] * 32 * S, p[1] / original[1] * 32 * S)


def squared_distance(p1, p2):
    return (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2


def target_transform(img_size, labels):
    if len(labels) > 4:
        print(f"OH NOOOOO there are too many labels: {labels}")

    out = torch.zeros((15, 2, 2))

    print(F"{img_size=} {labels=}")

    # Transform labels into resized space using img size
    unassigned = [(1, *into_resized(img_size, (top, left)),
                   *into_resized(img_size, (height, width)),
                   *(F.one_hot(torch.tensor(c - 1), num_classes=10)).numpy())
                  for (c, top, height, left, width) in labels]

    half_cell = 0.5 / S
    for x in range(S):
        for y in range(S):
            if len(unassigned) > 0:
                # Compute distance to cell center from unassigned labels
                cell_center = into_resized(
                    (1, 1), (x / S + half_cell, y / S + half_cell))

                distances = [
                    squared_distance(cell_center,
                                     (left + width / 2.0, top + height / 2.0))
                    for (_, top, left, height, width, _, _, _, _, _, _, _, _,
                         _, _) in unassigned
                ]

                # Assign the closest one to cell
                closest_i = np.argmin(distances)
                closest = unassigned[closest_i]
                print(f" assigning ({closest_i}) {closest} to {x} {y}")

                # TODO: There must be a smarter way to do this assignment
                for i in range(15):
                    out[i][x][y] = closest[i]

                del unassigned[closest_i]

    print(f"{out}")
    return out


svhn_train = SVHN(split='train',
                  transform=transform,
                  target_transform=target_transform)
svhn_dev = SVHN(split='dev',
                transform=transform,
                target_transform=target_transform)

train = DataLoader(svhn_train, batch_size=batch_size)
dev = DataLoader(svhn_dev)

img, y = svhn_train[0]

model = BaseModel()
opt = torch.optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=momentum)  # Stochastic Gradient Descent

fit(model, epochs, loss_func, opt, train, dev)

# print(f"{model.forward().shape=}")
