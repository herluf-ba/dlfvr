import torch 
import matplotlib.pyplot as plt

# Stolen from lab3 
def get_layer_data(model):
    #TODO: Why isn't activations working? 
    '''
    Examples usage: 
    layer_names, activations, gradients = get_layer_data(model)
    print('layer_names',layer_names)
    print('Activation list length:',len(activations))
    print('Activations shape layer 1:',activations[0].shape)
    print('Gradient list length:',len(gradients))
    print('Gradient shape layer 1:',gradients[0].shape)
    '''

    gradients = []
    layer_names = []
    weights = []

    with torch.no_grad():
        for name, param in model.named_parameters():
            # name format ex: encoded.0.weight
            param_type = name.split(".")[-1]
            if param.requires_grad and param_type.startswith('w'):
                layer_names.append(name)
                gradients.append(param.grad)
                weights.append(param)
                

        #activations = model.activations()
        activations = None  
    return layer_names, activations, gradients, weights


def get_layer_stats(x,absolute=False):
    '''
    Example usage:
        activation_mean, activation_std = get_layer_stats(activations)
        gradient_mean, gradient_std = get_layer_stats(gradients,absolute=True)

        print('activation_mean',activation_mean)
        print('activation_std',activation_std)
        print('gradient_mean',gradient_mean)
        print('gradient_std',gradient_std)
    '''    
    avg = []
    std = []
    for layer in range(len(x)):
        if absolute: # for gradient flow
          avg.append(x[layer].abs().mean().detach().cpu().numpy())
        else: # for activations
          avg.append(x[layer].mean().detach().cpu().numpy())

        std.append(x[layer].std().detach().cpu().numpy())
    
    return avg, std


def plot_hist(hs, layer_names, xrange=(-1,1),avg=None,sd=None):
    ''' 
    Example usage:
        print('Gradients:\n')
        plot_hist(gradients,xrange=None,avg=gradient_mean,sd=gradient_std)
        plt.show()

        print('Activations:\n')
        plot_hist(activations,xrange=None,avg=activation_mean,sd=activation_std)
        plt.show()
    '''
    plt.figure(figsize=(120,10))
    for layer in range(len(hs)):
        plt.subplot(1,len(hs),layer+1)
        activations = hs[layer].detach().cpu().numpy().flatten()
        plt.hist(activations, bins=20, range=xrange)

        title = layer_names[layer] 
        if avg:
          title += '\n' + "mean {0:.2f}".format(avg[layer])
        if sd:
          title += '\n' + "std {0:.4f}".format(sd[layer])

        plt.title(title)

def plot_grad_flow(named_parameters):
    '''
    Stolen from this guy: 
        https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/7
    '''
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def show_grad_flow():
    plt.show()
