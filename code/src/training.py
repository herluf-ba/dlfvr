import torch as t
import torch.nn as nn
import inspect
import numpy as np
from display import printProgressBar
<<<<<<< HEAD
from settings import batch_extract_classes
=======
from inspection import plot_grad_flow
>>>>>>> 9c04a80e34e1cead5d3906b3579f9919fb243523

def score_batch(model, loss_func, xb, yb, opt=None, logger=None, diagnoser=None, epoch=None):
    prediction = model.forward(xb)
    loss = loss_func(prediction, yb, logger=logger)

    if opt is not None:
        loss.backward()
        #plot_grad_flow(model.named_parameters())
        # Gotta diagnose right after backward - grad is calculated here
        if diagnoser: 
            diagnoser.diagnose_model(model, save_suffix=f'_e{epoch}') 
        opt.step()
        opt.zero_grad()

    if logger:
        ## Compute accuracy for classes
        target_classes = batch_extract_classes(yb).detach().numpy()
        pred_classes = batch_extract_classes(prediction).detach().numpy()
        true_positives = target_classes.argmax(axis=2) == pred_classes.argmax(
            axis=2)
        accuracy = true_positives.mean()
        logger.add_loss_item("accuracy", accuracy)

        ## TODO Compute accuracy for confidence

    return loss.item(), len(xb)


def fit(model, epochs, loss_func, opt, train_dl, valid_dl, device, logger):
    # Function in function? Mmmmm delicious spaghetti! 🤌
    def score(dl):
        ## Compute loss
        losses, nums = zip(*[
            score_batch(
                model, loss_func, xb.to(device), yb.to(device), logger=logger)
            for _, (xb, yb) in enumerate(dl)
        ])

        ## Save epoch log data to logger history
        logger.commit_epoch()

    batch_count = len(train_dl)

    for epoch in range(epochs):
        epoch_prefix = f"Epoch: {epoch + 1} / {epochs}"
        model.train()
        for batch_i, (xb, yb) in enumerate(train_dl):
            printProgressBar(batch_i, batch_count, prefix=epoch_prefix)
            xb, yb = xb.to(device), yb.to(device)
            # Diagnose on last batch of epoch
            diagnoser = logger if len(train_dl ) - 1 == batch_i else None
            score_batch(model, loss_func, xb, yb, opt, logger=None, diagnoser=diagnoser, epoch=epoch)

        printProgressBar(batch_count, batch_count, prefix=epoch_prefix)

        model.eval()
        with t.no_grad():

            print('Calculating validation loss')
            logger.set_mode('val')
            score(valid_dl)

            print('Calculating training loss ')
            logger.set_mode('train')
            score(train_dl)
