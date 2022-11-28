import numpy as np
import matplotlib.pyplot as plt


class Logger:
    mode = 'val'
    current_epoch = {}
    history = {}

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

        plt.show()
