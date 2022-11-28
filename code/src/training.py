import torch as t
import torch.nn as nn
import inspect
import numpy as np
from display import printProgressBar

def score_batch(model, loss_func, xb, yb, opt=None, logger=None, diagnoser=None, epoch=None):
    prediction = model.forward(xb)

    should_log = logger and 'logger' in inspect.getargspec(loss_func).args
    loss = loss_func(prediction, yb,
                     logger=logger) if should_log else loss_func(
                         prediction, yb)

    if opt is not None:
        loss.backward()
        # Gotta diagnose right after backward - grad is calculated here
        if diagnoser: 
            diagnoser.diagnose_model(model, save_suffix=f'_e{epoch}') 
        opt.step()
        opt.zero_grad()

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

        

