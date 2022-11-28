import torch as t
import torch.nn as nn
import inspect
import numpy as np
from display import printProgressBar
from inspection import get_layer_data, get_layer_stats, plot_hist
import matplotlib.pyplot as plt

def diagnose_model(model): 
    layer_names, activations, gradients = get_layer_data(model); 
    gradient_mean, gradient_std = get_layer_stats(gradients, absolute=True)
    print(f'{gradient_mean=}, {gradient_std=}')

    plot_hist(gradients,xrange=None,avg=gradient_mean,sd=gradient_std)
    plt.show()

    #plot_hist(activations,xrange=None,avg=activation_mean,sd=activation_std)
    #plt.show()


def score_batch(model, loss_func, xb, yb, opt=None, logger=None, save_diagnose_plots=False):
    prediction = model.forward(xb)

    should_log = logger and 'logger' in inspect.getargspec(loss_func).args
    loss = loss_func(prediction, yb,
                     logger=logger) if should_log else loss_func(
                         prediction, yb)

    if opt is not None:
        loss.backward()
        # Gotta diagnose right after backward - grad is calculated here
        if save_diagnose_plots: 
            diagnose_model(model) 
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
            score_batch(model, loss_func, xb, yb, opt, logger=None, save_diagnose_plots=batch_i == len(train_dl) - 1)

        printProgressBar(batch_count, batch_count, prefix=epoch_prefix)

        model.eval()
        with t.no_grad():

            print('Calculating validation loss')
            logger.set_mode('val')
            score(valid_dl)

            print('Calculating training loss ')
            logger.set_mode('train')
            score(train_dl)

        

