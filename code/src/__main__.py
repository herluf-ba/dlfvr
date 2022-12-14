import os
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
from settings import S, CONFIDENCE_THRESHOLD, MODELS, LOSS_FUNCTIONS, batch_extract_classes, batch_extract_confidence
from logger import Logger

if __name__ == '__main__':
    torch.manual_seed(42) # Set seed for reproduceability 
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
                           default=32)
    argparser.add_argument("-lr",
                           "--learning-rate",
                           help="Learning rate to use when training",
                           default=0.001)
    argparser.add_argument(
        "-mom",
        "--momentum",
        help="Momentum to use when training using gradient descent",
        default=0.0)
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
    argparser.add_argument(
        '-wi',
        "--weight-init",
        default=None,
        help="The method used for initializing weight of parametized layers"),
    argparser.add_argument(
        '-lrs', 
        '--learning-rate-scheduler', 
        help = "The learning rate scheduler to use for training.",
        default=None
    ), 
    argparser.add_argument(
            '--use-layer-specific-learning-rates', 
            help = 'Will set the learning rate of the bounding box decoder to 0.00043 and use the learning rate provided by -lr for the remaining', 
            action="store_true"),
    argparser.add_argument(
            '-a', 
            '--use-adam', 
            help = "Will change the optimizer from SGD to ADAM.",
            action="store_false"
            )

    args = argparser.parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    model_class = MODELS[args.model]
    model = model_class(weight_init=args.weight_init).to(device)

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
        split = args.split
        lrs = args.learning_rate_scheduler
        loss_func = LOSS_FUNCTIONS[
            args.
            loss_func]


        print("Training on device:", device)
        print(
            f"{learning_rate=}\n{momentum=}\n{batch_size=}\n{epochs=}\nloss_func={args.loss_func}\nlearning_rate_scheduler={lrs}"
        )

        train_split, test_split = (f'{split}_train', f'{split}_test')

        # Load data splits
        svhn_train = SVHN(split=train_split,
                          transform=transform,
                          target_transform=target_transform)

        svhn_test = SVHN(split=test_split,
                         transform=transform,
                         target_transform=target_transform)

        train = DataLoader(svhn_train, batch_size=batch_size)
        test = DataLoader(svhn_test)

        # Get model params
        model_params = None 
        if args.use_layer_specific_learning_rates: 
            bounding_box_learning_rate = 0.00043
            print(f'{bounding_box_learning_rate=}')
            model_params = [
                    {"params": model.encoded.parameters() }, 
                    {"params": model.confidence.parameters()}, 
                    {"params": model.classes.parameters()}, 
                    {"params": model.bounding_box.parameters(), 'lr': bounding_box_learning_rate}
            ]
        else: 
            model_params = model.parameters()

        # Choose between ADAM or SGD
        opt = None
        if args.use_adam: 
            print('Optimizer=Adam')
            opt = torch.optim.Adam(model_params, lr=learning_rate)
        else:
            print('Optimizer=SGD')
            opt = torch.optim.SGD(model_params,
                                  lr=learning_rate,
                                  momentum=momentum)

        # Setup learning rate scheduler (None ended up being used for experiments anyways)
        scheduler = None 
        assert lrs in ['step_lr', 'cos_wr', None], "Invalid learning rate scheduler. Should be one of: 'step_lr' or 'cos_wr' "
        if (lrs == 'step_lr'): 
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.95) # TODO find argument for gamma 
        elif (lrs == 'cos_wr'):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5) #TODO: find argument for T_0

        # Find path to save artifacts of training to
        fingerprint = f'{args.model}-{args.loss_func}-{split}-e_{epochs}-bs_{batch_size}-mom_{momentum}-lr_{learning_rate}-wi_{args.weight_init}-lrs_{lrs}_lslr-{args.use_layer_specific_learning_rates}'
        save_path = f'runs/{fingerprint}'
        i = 0
        while os.path.exists(save_path):
            i += 1
            save_path = f'runs/{fingerprint}_{i}'
        os.mkdir(save_path)
        os.mkdir(f'{save_path}/weights')

        # Setup logger & begin training!
        logger = Logger(save_path, model)
        fit(model, epochs, loss_func, opt, train, test, device, logger, lr_scheduler=scheduler)

        ## Save trained model
        if (args.save):
            print(f"Saving state dict to path: '{args.save}'")
            torch.save(model.state_dict(), args.save)
        

    ## Produce a predict if configured to do so
    predict_image_path = args.predict
    if (predict_image_path is not None):
        raw = read_image(predict_image_path)
        image = transform(raw).unsqueeze(dim=0).to(device)
        predictions = model.forward(image)
        # Move back to cpu for plotting
        predictions = predictions
        classes = batch_extract_classes(predictions)
        confidence = batch_extract_confidence(predictions)
        #print(classes[0].softmax(dim=1).argmax(dim=1))
        #print(confidence[0].sigmoid())
        
        # detach for numpy functions in libraries to work with tensors
        plot_img(raw, predictions[0].cpu().detach(), conf_threshold=CONFIDENCE_THRESHOLD)

