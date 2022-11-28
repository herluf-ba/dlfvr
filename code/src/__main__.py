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

from svhn import SVHN, transform, target_transform
from display import plot_img
from training import fit
from settings import S, MODELS, LOSS_FUNCTIONS, batch_extract_classes, batch_extract_confidence
from logger import Logger

if __name__ == '__main__':
    # Setup argument parser
    argparser = argparse.ArgumentParser(
        description='Model based on lab3 simple yolo implementation')
    argparser.add_argument("-sp",
                           "--split",
                           help="Controls which split is used.",
                           default='default')
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
        loss_func = LOSS_FUNCTIONS[
            args.
            loss_func]  #Hardcoded this while testing. Circular import bullshittery.

        print("Training on device:", device)
        print(
            f"{learning_rate=}\n{momentum=}\n{batch_size=}\n{epochs=}\nloss_func={args.loss_func}"
        )

        train_split, test_split = (f'{args.split}_train', f'{args.split}_test')

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

        logger = Logger()
        fit(model, epochs, loss_func, opt, train, test, device, logger)

        ## Save trained model
        if (args.save):
            print(f"Saving state dict to path: '{args.save}'")
            torch.save(model.state_dict(), args.save)

        logger.plot_loss_items(logger.history.keys(),
                               title=f'Loss over {args.epochs} epochs')
        logger.dump_to_csv()
        #logger.plot_loss_items(['Validation loss', 'Training loss'],
        #                       title=f"Loss over {args.epochs} epochs")
        #logger.plot_loss_items(['Validation IoU', 'Training IoU'],
        #                       title='IoU over {args.epochs} epochs')

    ## Produce a predict if configured to do so
    predict_image_path = args.predict
    if (predict_image_path is not None):
        image = transform(read_image(predict_image_path))
        image = image.unsqueeze(dim=0).to(device)
        labels = model.forward(image)

        classes = batch_extract_classes(labels)
        confidences = batch_extract_confidence(labels)

        # Move back to cpu for plotting
        image = image.cpu()
        labels = labels.cpu()

        # detach for numpy functions in libraries to work with tensors
        plot_img(image[0].detach(), labels[0].detach(), conf_threshold=0.0)
