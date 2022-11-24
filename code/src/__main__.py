# TODO: Do we somehow normalize labels? I do not think we should do that.
# TODO: Get the predicted class in "plot_img" to label each bounding box
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
from torchvision.transforms import Compose, Grayscale, Resize, Lambda, ToTensor
from torchvision.io import read_image
import numpy as np

from svhn import SVHN
from display import plot_loss_history, plot_img
from training import fit
from base_model import BaseModel

S = 2
MODELS = {"base": BaseModel, "skipper": None}
LOSS_FUNCTIONS = {"mse": F.mse_loss, "exp_mse": None}


def into_resized(original, p):
    return (p[0] / original[0] * 32 * S, p[1] / original[1] * 32 * S)


def squared_distance(p1, p2):
    return (p2[0] - p1[0])**2 + (p2[1] - p1[1])**2


def target_transform(img_size, labels):
    # stfu computer if len(labels) > 4:
    #    print(f"OH NOOOOO there are too many labels: {labels}")

    out = torch.zeros((15, 2, 2))

    # Transform labels into resized space using img size
    unassigned = [(1, *into_resized(img_size, (top, left)),
                   *into_resized(img_size, (height, width)),
                   *(F.one_hot(torch.tensor(c - 1), num_classes=10)).numpy())
                  for (c, left, top, width, height) in labels]

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

                # TODO: There must be a smarter way to do this assignment
                for i in range(15):
                    out[i][x][y] = closest[i]

                del unassigned[closest_i]

    return out


def train_base_model(model,
                     device,
                     transform,
                     split,
                     learning_rate,
                     momentum,
                     batch_size,
                     epochs,
                     loss_func,
                     save_path=None):
    train_split, test_split = (
        f'{split}_train', f'{split}_test') if split is not None else ('train',
                                                                      'test')
    # Load data splits
    svhn_train = SVHN(split=train_split,
                      transform=transform,
                      target_transform=target_transform)

    svhn_test = SVHN(split=test_split,
                     transform=transform,
                     target_transform=target_transform)

    train = DataLoader(svhn_train, batch_size=batch_size)
    test = DataLoader(svhn_test)

    # Train model
    opt = torch.optim.SGD(model.parameters(),
                          lr=learning_rate,
                          momentum=momentum)

    return fit(model, epochs, loss_func, opt, train, test, device)


if __name__ == '__main__':
    # Setup argument parser
    argparser = argparse.ArgumentParser(
        description='Model based on lab3 simple yolo implementation')
    argparser.add_argument("-sp",
                           "--split",
                           help="Controls which split is used.",
                           default=None)
    argparser.add_argument(
        "-l",
        "--load",
        help=
        "Path to a saved state dictionary the model should be initialized with.",
        default=None)
    argparser.add_argument(
        "-s",
        "--save",
        help=
        "Location where the state dictionary of the model should be saved.",
        default=None)
    argparser.add_argument("-e",
                           "--epochs",
                           help="Number of epochs to run fit for.",
                           default=1)
    argparser.add_argument("-bs",
                           "--batch-size",
                           help="Batch size to use when training",
                           default=64)
    argparser.add_argument("-lr",
                           "--learning-rate",
                           help="Learning rate to use when training",
                           default=0.001)
    argparser.add_argument(
        "-mom",
        "--momentum",
        help="Momentum to use when training using gradient descent",
        default=0.9)
    argparser.add_argument("-p", "--predict", help="Predict a single image")
    argparser.add_argument('-t',
                           "--train",
                           help="Train a new base model.",
                           action="store_true")
    argparser.add_argument(
        "-lf",
        "--loss-func",
        help=
        f'Loss function used for training [{", ".join(LOSS_FUNCTIONS.keys())}]',
        default="mse")
    argparser.add_argument(
        "-m",
        "--model",
        help=f'CNN Model to be used [{", ".join(MODELS.keys())}]',
        default="base")

    args = argparser.parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = MODELS[args.model]().to(device)

    transform = Compose([
        Lambda(lambda img: img / 255),
        Grayscale(),
        Resize((32 * S, 32 * S)),
    ])

    # Load weights if given
    load_path = args.load
    if (load_path is not None):
        print(f"Loading state dict from path: '{load_path}'")
        model.load_state_dict(torch.load(load_path, map_location=device))

    if (args.train):
        #Extract hyper parameters for learning
        learning_rate = float(args.learning_rate)
        momentum = float(args.momentum)
        batch_size = int(args.batch_size)
        epochs = int(args.epochs)
        loss_func = LOSS_FUNCTIONS[args.loss_func]

        print("Training on device:", device)
        print(
            f"{learning_rate=}\n{momentum=}\n{batch_size=}\n{epochs=}\nloss_func={args.loss_func}"
        )

        train_loss_hist, val_loss_hist = train_base_model(
            model, device, transform, learning_rate, momentum, batch_size,
            epochs, loss_func, args.split)

        ## Save trained model
        if (args.save_path is not None):
            print(f"Saving state dict to path: '{args.save_path}'")
            torch.save(model.state_dict(), args.save_path)

        plot_loss_history(train_loss_hist, val_loss_hist)

    ## Produce a predict if configured to do so
    predict_image_path = args.predict
    if (predict_image_path is not None):
        image = transform(read_image(predict_image_path))
        image = image.unsqueeze(dim=0)  # Wraps it in a "batch"
        labels = model.forward(image)

        # detach for numpy functions in libraries to work with tensors
        plot_img(image[0].detach(), labels[0].detach(), conf_threshold=0.0)
