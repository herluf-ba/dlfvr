import pandas
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
    epoch_gradients = []

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
        self.epoch_gradients.append(grads)
 
    def collect_model_data(self):
        self.collect_gradients()

    def commit_epoch(self):
        ## Save model histograms before clearing collected gradient data
        self.weight_histogram()
        self.gradient_histogram()

        ## Flush mean of metric to csv file
        if not self.f_metrics:
            self.f_metrics = open(f'{self.save_path}/metrics.csv', 'a+')
            self.f_metrics.write("epoch," + ",".join(self.epoch_metrics.keys()) + '\n')

        line = [np.mean(items) for items in self.epoch_metrics.values()]
        self.f_metrics.write(f'{self.epoch},' + ','.join(map(str, line)) + '\n')
        self.f_metrics.flush()

        ## Flush average gradients to csv file 
        if not self.f_avg_gradients:
            self.f_avg_gradients = open(f'{self.save_path}/avg_gradients.csv', 'a+')
            self.f_avg_gradients.write("epoch,batch," + ",".join(self.weight_layers) + '\n')

        averages_per_batch = [[layer_grads.mean().item() for layer_grads in batch_grads] for batch_grads in self.epoch_gradients]
        for b, averages in enumerate(averages_per_batch):
            self.f_avg_gradients.write(f'{self.epoch},{b},' + ",".join(map(str, averages)) + "\n")
        self.f_avg_gradients.flush()

        ## Clear for next epoch
        self.epoch_metrics = {}
        self.epoch_gradients = []
        self.epoch += 1
                
    ## PLOTTERS
    def plot_metrics(self, title=""):
        self.f_metrics.seek(0)
        data = pandas.read_csv(self.f_metrics)
        names = data.axes[1][:1]

        plt.title(title)
        for name in names:
            items = data.get(name)
            epochs = list(range(len(items)))
            plt.plot(epochs, items, label=name)

        plt.legend()
        plt.savefig(f'{self.save_path}/metrics.png')
        plt.close('all')  # Clear for future plotting

    def plot_gradient_flow(self, include_decoders = ['confidence', 'bounding_box', 'classes']):
        self.f_avg_gradients.seek(0)
        data = pandas.read_csv(self.f_avg_gradients)

        plt.xlabel("Layers")
        plt.ylabel("Avg. gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.figure(figsize=(10, 10), dpi=100)
        
        weight_mask = [w for w in self.weight_layers if 'encoded' in w or w.split('.')[0] in include_decoders]
        included_weights = [w.replace('.weight', '') for w in weight_mask]
        num_weights = len(included_weights)
        plt.hlines(0, 0, num_weights + 1, linewidth=1, color="k" )
        plt.xticks(range(0, num_weights), included_weights, rotation="vertical")
        plt.xlim(xmin=0, xmax=num_weights - 1)

        for b, avg_grads in data.iterrows():
            plt.plot(avg_grads[weight_mask], alpha=0.3, color="b")
        
        plt.savefig(f'{self.save_path}/gradient_flow_{"_".join(include_decoders)}.png')
        plt.close('all')  # Clear for future plotting

    def weight_histogram(self):
        weights = [
            p.detach().cpu().numpy() for name, p, in self.model.named_parameters()
            if name in self.weight_layers
        ]
        means = [ws.mean().item() for ws in weights]
        stds = [ws.std().item() for ws in weights]

        # Plot weights
        plt.figure(figsize=(120,10))
        for i in range(len(weights)):
            plt.subplot(1, len(weights), i + 1)
            plt.hist(weights[i].flatten(), bins=20)
            mean = '{0:0.2f}'.format(means[i])
            std = '{0:0.4f}'.format(stds[i])
            name = self.weight_layers[i].replace(".weight", "") 
            plt.title(f'{name} mean: {mean} std: {std}')

        plt.savefig(f'{self.save_path}/weights_histogram_e{self.epoch}.png')
        plt.close('all')  # Clear for future plotting
    
    def gradient_histogram(self):
        gradients = self.epoch_gradients[-1]
        means = [gs.abs().mean().item() for gs in gradients]
        stds = [gs.std().item() for gs in gradients]

        # Plot gradients
        plt.figure(figsize=(120,10))
        for i in range(len(gradients)):
            plt.subplot(1, len(gradients), i + 1)
            plt.hist(gradients[i], bins=20)
            mean = '{0:0.2f}'.format(means[i])
            std = '{0:0.4f}'.format(stds[i])
            name = self.weight_layers[i].replace(".weight", "") 
            plt.title(f'{name} mean: {mean} std: {std}')

        plt.savefig(f'{self.save_path}/gradients_histogram_e{self.epoch}.png')
        plt.close('all')  # Clear for future plotting

