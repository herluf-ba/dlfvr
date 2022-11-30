import numpy as np
import matplotlib.pyplot as plt
from inspection import get_layer_data, get_layer_stats, plot_hist


class Logger:
    mode = 'val'
    model = None
    epoch = 0

    # Stuff for tracking model data
    f_avg_gradients = None
    weight_layers = []
    weights_per_batch = []
    gradients_per_batch = []

    # Stuff for tracking metrics (loss, accuracy, etc)
    f_metrics = None
    epoch_metrics = {}

    def __init__(self, save_path, model):
        self.save_path = save_path
        self.model = model

        ## Write down the layer names that requires gradients
        ## TODO: Why dont we write down bias?
        for name, p in model.named_parameters():
            is_weight = name.split('.')[-1].startswith('w')
            if p.requires_grad and is_weight:
                self.weight_layers.append(name)

        ## Create csv files for storing tracked data
        self.f_avg_gradients = open(f'{self.save_path}/avg_gradients.csv', 'a')
        self.f_avg_gradients.write("batch," + ",".join(self.weight_layers) + '\n')


    def set_mode(self, mode):
        assert mode in ['val', 'train']
        self.mode = mode

    ## DATA COLLECTION
    def add_metric(self, _name, item):

        prefix = "Validation " if self.mode == 'val' else "Train "
        name = f'{prefix}{_name}'

        if name in self.epoch_metrics.keys():
            self.epoch_metrics[name] = np.append(self.epoch_metrics[name],
                                                 item)
        else:
            self.epoch_metrics[name] = np.array([item])
    
    def collect_gradients(self):
        grads = [
            p.grad.abs().flatten().cpu() for name, p in self.model.named_parameters()
            if name in self.weight_layers
        ]

        self.gradients_per_batch.append(grads)

    def collect_weights(self):
        weights = [
            p for name, p, in self.model.named_parameters()
            if name in self.weight_layers
        ]
        self.weights_per_batch.append(weights)

    def collect_model_data(self):
        self.collect_gradients()
        self.collect_weights()

    def commit_epoch(self):
        if not self.f_metrics:
            self.f_metrics = open(f'{self.save_path}/metrics.csv', 'a')
            self.f_metrics.write("epoch," + ",".join(self.epoch_metrics.keys()) + '\n')

        ## Compute the mean of items and record as metric for that epoch
        line = [np.mean(items) for items in self.epoch_metrics.values()]
        self.f_metrics.write(f'{self.epoch},' + ','.join(map(str, line)) + '\n')
        self.f_metrics.flush()

        ## Clear for next epoch
        self.epoch_metrics = {}
        self.epoch += 1

    ## HELPERS
    def get_avg_gradients_per_batch(self):
        return [[layer_grads.mean().item() for layer_grads in batch_grads] for batch_grads in self.gradients_per_batch]
                
    ## PLOTTERS
    def plot_metrics(self, names, title=""):
        plt.title(title)

        for name in names:
            items = self.metrics[name]
            epochs = list(range(len(items)))
            plt.plot(epochs, items, label=name)

        if len(names) > 1:
            plt.legend()

        plt.savefig(f'{self.save_path}/{"-".join(names)}.png')
        plt.close('all')  # Clear for future plotting

    def plot_gradient_flow(self, include_decoders = ['confidence', 'bounding_box', 'classes']):
        plt.xlabel("Layers")
        plt.ylabel("Avg. gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.figure(figsize=(10, 10), dpi=100)
        
        weight_mask = [i for i, w in enumerate(self.weight_layers) if 'encoded' in w or w.split('.')[0] in include_decoders]
        included_weights = [self.weight_layers[i].replace('.weight', '') for i in weight_mask]
        num_weights = len(included_weights)
        plt.hlines(0, 0, num_weights + 1, linewidth=1, color="k" )
        plt.xticks(range(0, num_weights), included_weights, rotation="vertical")
        plt.xlim(xmin=0, xmax=num_weights - 1)

        for b, avg_grads in enumerate(self.get_avg_gradients_per_batch()):
            included_avg_grads = [avg_grads[i] for i in weight_mask]
            plt.plot(included_avg_grads, alpha=0.3, color="b")
        
        plt.savefig(f'{self.save_path}/gradient_flow_{"_".join(include_decoders)}.png')
        plt.close('all')  # Clear for future plotting

    def diagnose_model(self, model, save_suffix=''):
        weight_layers, activations, gradients, weights = get_layer_data(model)
        gradient_mean, gradient_std = get_layer_stats(gradients, absolute=True)
        weights_mean, weights_std = get_layer_stats(weights)

        # Add save suffix to each layer name
        weight_layers = [
            f'{layer_name}{save_suffix}' for layer_name in weight_layers
        ]

        # Plot weights
        plot_hist(weights,
                  weight_layers,
                  xrange=None,
                  avg=gradient_mean,
                  sd=gradient_std)
        plt.savefig(f'{self.save_path}/weights_histogram{save_suffix}.png')
        plt.close('all')  # Clear for future plotting

        # Plot gradients
        gradient_weight_layers = [
            layer_name.replace('weight', 'gradient')
            for layer_name in weight_layers
        ]  # encoded.06.weights -> encoded.06.gradients
        plot_hist(gradients,
                  gradient_weight_layers,
                  xrange=None,
                  avg=gradient_mean,
                  sd=gradient_std)
        plt.savefig(f'{self.save_path}/gradients_histogram{save_suffix}.png')
        plt.close('all')  # Clear for future plotting

    ## CSV DUMPERS
    def dump_avg_gradients_to_csv(self):
        with open(f'{self.save_path}/avg_gradients.csv', "w") as csv:
            averages_per_batch = self.get_avg_gradients_per_batch()
            for b, averages in enumerate(averages_per_batch):
                csv.write(f'{b},' + ",".join(map(str, averages)) + "\n")
