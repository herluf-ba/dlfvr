import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

def metrics_from_cm(confusion_matrix):
    tp = np.diagonal(confusion_matrix).sum() # sum of diagonal
    fp = np.triu(confusion_matrix, -1).sum() # sum of all cells under diagonal
    fn = np.tril(confusion_matrix, 1).sum()  # sum of cells above diagonal
    total = tp + fp + fn
    acc = tp / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return [acc, precision, recall]

def plot_cm(cm, path):
    normalized_cm = cm / np.sum(cm)
    num_classes = len(cm)
    fig, ax = plt.subplots(figsize=(min([num_classes-1, 2]), min([num_classes-1, 2])), dpi=100)
    ax.matshow(normalized_cm)
    for t in range(num_classes):
        for p in range(num_classes):
            v = normalized_cm[t][p]
            ax.text(p, t, '{0:0.2f}'.format(v), ha='center', va='center', fontsize='small', color='#ffffff' if v < 0.5 else '#252525')
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    ax.set_title("Confusion matrix")
    ax.set_ylabel("True")
    ax.set_xlabel('Predicted')
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticks(np.arange(num_classes), )
    plt.savefig(path)
    plt.close('all')  # Clear for future plotting

def plot_accuracy(train, val, path):
    epochs = list(range(1, len(val) + 1))
    plt.plot(epochs, val, label='validation')
    plt.plot(epochs, train, label='train')
    plt.title("Accuracy")
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.savefig(path)
    plt.close('all')  # Clear for future plotting

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

    ## Confusion matrix for classes and confidence
    train_classes_cm = np.zeros((10, 10))
    train_confidence_cm = np.zeros((2, 2))
    val_classes_cm = np.zeros((10, 10))
    val_confidence_cm = np.zeros((2, 2))
    # Note: These are all computed from confusion matricies
    default_metric_names = [
        'Train classes accuracy',
        'Train classes precision',
        'Train classes recall',
        'Validation classes accuracy',
        'Validation classes precision',
        'Validation classes recall',
        'Train confidence accuracy',
        'Train confidence precision',
        'Train confidence recall',
        'Validation confidence accuracy',
        'Validation confidence precision',
        'Validation confidence recall',
    ]

    def __init__(self, save_path, model):
        self.save_path = save_path
        self.model = model

        # Make dirs for saving artifacts
        os.mkdir(f'{self.save_path}/csv')
        os.mkdir(f'{self.save_path}/metrics')
        os.mkdir(f'{self.save_path}/gradient_flow')
        os.mkdir(f'{self.save_path}/gradient_histograms')
        os.mkdir(f'{self.save_path}/weight_histograms')
        os.mkdir(f'{self.save_path}/classes_confusion_matricies')
        os.mkdir(f'{self.save_path}/confidence_confusion_matricies')

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
    def add_loss_item(self, _name, item):
        prefix = "Validation " if self.mode == 'val' else "Train "
        name = f'{prefix}{_name} loss'
        if name in self.epoch_metrics.keys():
            self.epoch_metrics[name] = np.append(self.epoch_metrics[name], item)
        else:
            self.epoch_metrics[name] = np.array([item])

    def collect_classes_metrics(self, true, predicted):
        cm = self.train_classes_cm if self.mode == 'train' else self.val_classes_cm
        for (t, p) in zip(true, predicted):
            cm[t.argmax().item()][p.argmax().item()] += 1
    
    def collect_confidence_metrics(self, true, predicted):
        cm = self.train_confidence_cm if self.mode == 'train' else self.val_confidence_cm
        for (t, p) in zip(true.flatten(), predicted.flatten()):
            cm[int(t.item())][int(p.item())] += 1

    def collect_gradients(self):
        self.epoch_gradients.append([
            p.grad.abs().flatten().cpu() for name, p in self.model.named_parameters()
            if name in self.weight_layers
        ])

    def commit_epoch(self):
        # Use classes and confidence confusion matricies to compute accuracy, precision and recall for both train and validation
        default_metrics = [
            *metrics_from_cm(self.train_classes_cm),
            *metrics_from_cm(self.val_classes_cm),
            *metrics_from_cm(self.train_confidence_cm),
            *metrics_from_cm(self.val_confidence_cm)
        ]

        ## Flush metrics to csv file
        if not self.f_metrics:
            self.f_metrics = open(f'{self.save_path}/csv/metrics.csv', 'a+')
            self.f_metrics.write("epoch," + ",".join([*self.default_metric_names, *self.epoch_metrics.keys()]) + '\n')

        mean_metrics = [np.mean(items) for items in self.epoch_metrics.values()]
        self.f_metrics.write(f'{self.epoch},' + ','.join(map(str, [*default_metrics, *mean_metrics])) + '\n')
        self.f_metrics.flush()

        ## Flush average gradients to csv file 
        if not self.f_avg_gradients:
            self.f_avg_gradients = open(f'{self.save_path}/csv/avg_gradients.csv', 'a+')
            self.f_avg_gradients.write("epoch,batch," + ",".join(self.weight_layers) + '\n')

        averages_per_batch = [[layer_grads.mean().item() for layer_grads in batch_grads] for batch_grads in self.epoch_gradients]
        for b, averages in enumerate(averages_per_batch):
            self.f_avg_gradients.write(f'{self.epoch},{b},' + ",".join(map(str, averages)) + "\n")
        self.f_avg_gradients.flush()

        ## Produce plots for current epoch
        self.weight_histogram()
        self.gradient_histogram()
        self.plot_classes_cm()
        self.plot_confidence_cm()
        self.plot_classes_accuracy()
        self.plot_confidence_accuracy()
        self.plot_loss()
        self.plot_loss_items()
        self.plot_gradient_flow(include_decoders=['confidence'])
        self.plot_gradient_flow(include_decoders=['bounding_box'])
        self.plot_gradient_flow(include_decoders=['classes'])

        ## Clear for next epoch
        self.epoch_metrics = {}
        self.epoch_gradients = []
        train_classes_cm = np.zeros((10, 10))
        train_confidence_cm = np.zeros((2, 2))
        val_classes_cm = np.zeros((10, 10))
        val_confidence_cm = np.zeros((2, 2))
        self.epoch += 1
    

    ## PLOTTERS
    def plot_loss(self):
        self.f_metrics.seek(0)
        data = pandas.read_csv(self.f_metrics)
        names = [n for n in data.axes[1][1:] if 'loss' in n]
        val_items = np.array([data.get(n) for n in names if 'Validation' in n])
        train_items = np.array([data.get(n) for n in names if 'Train' in n])
        plt.title('Loss')
        epochs = list(range(len(val_items)))
        plt.plot(epochs, np.sum(train_items, axis=1), label='train')
        plt.plot(epochs, np.sum(val_items, axis=1), label='validation')
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.savefig(f'{self.save_path}/metrics/loss.png')
        plt.close('all')  # Clear for future plotting

    def plot_loss_items(self):
        self.f_metrics.seek(0)
        data = pandas.read_csv(self.f_metrics)
        names = [n for n in data.axes[1][1:] if 'loss' in n]
        plt.title('Loss items')
        for name in names:
            items = data.get(name)
            epochs = list(range(len(items)))
            plt.plot(epochs, items, label=name)

        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel('loss')
        plt.savefig(f'{self.save_path}/metrics/loss_items.png')
        plt.close('all')  # Clear for future plotting

    def plot_classes_accuracy(self, title=""):
        self.f_metrics.seek(0)
        data = pandas.read_csv(self.f_metrics)
        val = data.get('Validation classes accuracy')
        train = data.get("Train classes accuracy")
        path = f'{self.save_path}/metrics/classes_accuracy.png'
        plot_accuracy(train, val, path)

    def plot_confidence_accuracy(self, title=""):
        self.f_metrics.seek(0)
        data = pandas.read_csv(self.f_metrics)
        val = data.get("Validation confidence accuracy")
        train = data.get("Train confidence accuracy")
        path = f'{self.save_path}/metrics/confidence_accuracy.png'
        plot_accuracy(train, val, path)

    def plot_classes_cm(self):
        plot_cm(self.train_classes_cm, f'{self.save_path}/classes_confusion_matricies/train_e{self.epoch}.png')
        plot_cm(self.val_classes_cm, f'{self.save_path}/classes_confusion_matricies/val_e{self.epoch}.png')

    def plot_confidence_cm(self):
        plot_cm(self.train_confidence_cm, f'{self.save_path}/confidence_confusion_matricies/train_e{self.epoch}.png')
        plot_cm(self.val_confidence_cm, f'{self.save_path}/confidence_confusion_matricies/val_e{self.epoch}.png')

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
        
        plt.savefig(f'{self.save_path}/gradient_flow/{"_".join(include_decoders)}.png')
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

        plt.savefig(f'{self.save_path}/weight_histograms/e{self.epoch}.png')
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

        plt.savefig(f'{self.save_path}/gradient_histograms/e{self.epoch}.png')
        plt.close('all')  # Clear for future plotting

