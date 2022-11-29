import numpy as np
import matplotlib.pyplot as plt
from inspection import get_layer_data, get_layer_stats, plot_hist


class Logger:
    mode = 'val'
    current_epoch = {}
    history = {}

    def __init__(self, save_path): 
        self.save_path = save_path

    def set_mode(self, mode):
        assert mode in ['val', 'train']
        self.mode = mode

    def add_loss_item(self, _name, item):
        prefix = "Validation " if self.mode is 'val' else "Train "
        name = f'{prefix}{_name}'
        
        if name in self.current_epoch.keys():
            self.current_epoch[name] = np.append(self.current_epoch[name],
                                                 item)
        else:
            self.current_epoch[name] = np.array([item])

    def commit_epoch(self):
        prev_len = -1
        for (name, items) in self.current_epoch.items():
            ## Check that all items have the same length
            assert prev_len == -1 or prev_len == len(items)
            prev_len = len(items)

            ## Compute the mean of items and record as metric for that epoch
            mean = np.mean(items)
            if name in self.history.keys():
                self.history[name] = np.append(self.history[name], mean)
            else:
                self.history[name] = np.array([mean])

        ## Clear for next epoch
        self.current_epoch = {}

    def plot_loss_items(self, names, title=""):
        plt.title(title)

        for name in names:
            items = self.history[name]
            epochs = list(range(len(items)))
            plt.plot(epochs, items, label=name)

        if len(names) > 1:
            plt.legend()

        plt.savefig(f'{self.save_path}/loss.png')
        plt.close('all') # Clear for future plotting

    def diagnose_model(self, model, save_suffix=''): 
        layer_names, activations, gradients, weights = get_layer_data(model); 
        gradient_mean, gradient_std = get_layer_stats(gradients, absolute=True)
        weights_mean, weights_std = get_layer_stats(weights)
        
        # Add save suffix to each layer name
        layer_names = [f'{layer_name}{save_suffix}' for layer_name in layer_names]
        
        # Plot weights
        plot_hist(weights, layer_names, xrange=None,avg=gradient_mean,sd=gradient_std)
        plt.savefig(f'{self.save_path}/weights_histogram{save_suffix}.png')
        plt.close('all') # Clear for future plotting

        # Plot gradients
        gradient_layer_names = [layer_name.replace('weight','gradient') for layer_name in layer_names] # encoded.06.weights -> encoded.06.gradients
        plot_hist(gradients, gradient_layer_names, xrange=None,avg=gradient_mean,sd=gradient_std)
        plt.savefig(f'{self.save_path}/gradients_histogram{save_suffix}.png')
        plt.close('all') # Clear for future plotting


