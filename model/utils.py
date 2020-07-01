import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch
import pandas as pd
from GPdata_sampler import NPRegressionDescription

def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.

    Args:
        model (:class:`nn.Sequential`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model

def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.

    Args:
        layer (:class:`nn.Module`): Single dense or convolutional layer from
            :mod:`torch.nn`.

    Returns:
        :class:`nn.Module`: Single dense or convolutional layer with
            initialized weights.
    """
    nn.init.xavier_normal_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 1e-3)

def compute_loss(mean, var, y_target):
    dist = Normal(loc=mean, scale=var)
    log_prob = dist.log_prob(y_target)
    loss = - torch.mean(log_prob)
    return loss

def comput_kl_loss(prior, poster):
    div = kl_divergence(poster, prior)
    div = torch.mean(div, dim=0).sum()
    return div

def to_numpy(var):
    return var.detach().cpu().numpy()

def to_tensor(var):
    return torch.tensor(var,dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

def save_plot_data(data_test, kernel):
    data = data_test.generate_curves(include_context=False)
    (x_context, y_context), x_target = data.query
    y_target = data.y_target
    df = pd.DataFrame()
    df['x_context'] = to_numpy(x_context[0].squeeze())
    df['y_context'] = to_numpy(y_context[0].squeeze())
    df.to_csv('saved_fig/plot_context_'+kernel+'.csv', index=False)
    df = pd.DataFrame()
    df['x_target'] = to_numpy(x_target[0].squeeze())
    df['y_target'] = to_numpy(y_target[0].squeeze())
    df.to_csv('saved_fig/plot_target_' + kernel + '.csv', index=False)
    return data

def load_plot_data(kernel):
    context = pd.read_csv('saved_fig/plot_context_'+kernel+'.csv')
    target = pd.read_csv('saved_fig/plot_target_' + kernel + '.csv')
    query = (to_tensor(context['x_context']), to_tensor(context['y_context'])), to_tensor(target['x_target'])
    y_target = to_tensor(target['y_target'])
    return NPRegressionDescription(query=query, y_target=y_target, \
                                   num_context_points=context.shape[0], \
                                   num_total_points=target.shape[0]+context.shape[0])
