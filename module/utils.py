import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch
import pandas as pd
from data.GP_data_sampler import NPRegressionDescription

def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential module with Glorot
    initialization.

    Args:
        model (:class:`nn.Sequential`): Container for module.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.

    Returns:
        (nn.Sequential): module with initialized weights
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
    var = torch.tensor(var,dtype=torch.float32).unsqueeze(0)
    return var if len(var.shape) > 2 else var.unsqueeze(-1)

def save_plot_data(data_test, kernel):
    if kernel in ['MNIST', 'SVHN', 'celebA']: # image datesets
        img, _ = next(iter(data_test))
        context_mask, target_mask = generate_mask(img)
        x_context, y_context, x_target, y_target = img_mask_to_np_input(img, context_mask, target_mask, \
                                                                        include_context=False)
        query = (x_context, y_context), x_target
        data = NPRegressionDescription(query=query, y_target=y_target, \
                                num_context_points=x_context.shape[1], \
                                num_total_points=x_target.shape[1] + x_context.shape[1])
    else: # GP datasets
        data = data_test.generate_curves(include_context=False)
        (x_context, y_context), x_target = data.query
        y_target = data.y_target
    context = to_numpy(torch.cat([x_context[0], y_context[0]], dim=-1))
    # print ("context data shape:", context.shape)
    df = pd.DataFrame(context)
    df.to_csv('saved_fig/csv/plot_context_'+kernel+'.csv', index=False)
    target = to_numpy(torch.cat([x_target[0],y_target[0]], dim=-1))
    # print("target data shape:", target.shape)
    df = pd.DataFrame(target)
    df.to_csv('saved_fig/csv/plot_target_' + kernel + '.csv', index=False)
    return data

def load_plot_data(kernel):
    if kernel in ['MNIST', 'SVHN', 'celebA']:  # image datesets
        context = pd.read_csv('saved_fig/csv/plot_context_' + kernel + '.csv').values
        target = pd.read_csv('saved_fig/csv/plot_target_' + kernel + '.csv').values
        x_context = context[:, :2]
        y_context = context[:, 2:]
        x_target = target[:, :2]
        y_target = target[:, 2:]
        query = (to_tensor(x_context), to_tensor(y_context)), to_tensor(x_target)
        y_target = to_tensor(y_target)
    else: # GP datasets
        context = pd.read_csv('saved_fig/csv/plot_context_'+kernel+'.csv')
        target = pd.read_csv('saved_fig/csv/plot_target_' + kernel + '.csv')
        query = (to_tensor(context['x_context']), to_tensor(context['y_context'])), to_tensor(target['x_target'])
        y_target = to_tensor(target['y_target'])
    return NPRegressionDescription(query=query, y_target=y_target, \
                                   num_context_points=context.shape[0], \
                                   num_total_points=target.shape[0]+context.shape[0])

def img_mask_to_np_input(img, context_mask, target_mask, include_context = False, normalize=True):
    """
    Given an image and two masks, return x and y tensors expected by Neural
    Process. Specifically, x will contain indices of unmasked points, e.g.
    [[1, 0], [23, 14], [24, 19]] and y will contain the corresponding pixel
    intensities, e.g. [[0.2], [0.73], [0.12]] for grayscale or
    [[0.82, 0.71, 0.5], [0.42, 0.33, 0.81], [0.21, 0.23, 0.32]] for RGB.

    Parameters
    ----------
    img : torch.Tensor
        Shape (N, C, H, W). Pixel intensities should be in [0, 1]

    context_mask : torch.Tensor
        Context mask where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    target_mask : torch.Tensor
        Target mask where 0 corresponds to masked pixel and 1 to a visible
        pixel. Shape (N, H, W). Note the number of unmasked pixels must be the
        SAME for every mask in batch.

    normalize : bool
        If true normalizes pixel locations x to [-1, 1]
    """
    batch_size, num_channels, height, width = img.size()
    # Create a mask which matches exactly with image size which will be used to
    # extract pixel intensities
    context_mask_img_size = context_mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    target_mask_img_size = target_mask.unsqueeze(1).repeat(1, num_channels, 1, 1)
    # Number of points corresponds to number of visible pixels in mask, i.e. sum
    # of non zero indices in a mask (here we assume every mask has same number
    # of visible pixels)
    num_context_points = context_mask[0].nonzero().size(0)
    num_target_points  = target_mask[0].nonzero().size(0)

    # Compute non zero indices
    # Shape (num_nonzeros, 3), where each row contains index of batch, height and width of nonzero
    context_nonzero_idx = context_mask.nonzero()
    target_nonzero_idx = target_mask.nonzero()

    # The x tensor for Neural Processes contains (height, width) indices, i.e.
    # 1st and 2nd indices of nonzero_idx (in zero based indexing)
    x_context = context_nonzero_idx[:, 1:].view(batch_size, num_context_points, 2).float()
    # The y tensor for Neural Processes contains the values of non zero pixels
    y_context = img[context_mask_img_size].view(batch_size, num_channels, num_context_points)
    # Ensure correct shape, i.e. (batch_size, num_points, num_channels)
    y_context = y_context.permute(0, 2, 1)

    x_target = target_nonzero_idx[:, 1:].view(batch_size, num_target_points, 2).float()
    y_target = img[target_mask_img_size].view(batch_size, num_channels, num_target_points)
    y_target = y_target.permute(0, 2, 1)

    if normalize:
        # TODO: make this separate for height and width for non square image
        # Normalize x to [-1, 1]
        x_context = (x_context - float(height) / 2) / (float(height) / 2)
        x_target = (x_target - float(height) / 2) / (float(height) / 2)

    if include_context:
        x_target = torch.cat([x_target, x_context], dim=1)
        y_target = torch.cat([y_target, y_context], dim=1)
    return x_context, y_context, x_target, y_target

def generate_mask(img):
    """
    return a context mask and a target mask
    Args:
        img: input images, shape: [B, C, H, W]
    Returns:
        context mask, shape: [B, H, W]
        target mask, shape: [B, H, W]
    """
    batch_size = img.size(0)
    n_total = img.size(2) * img.size(3)
    num_context = int(torch.empty(1).uniform_(n_total / 100, n_total / 2).item())
    # num_extra_target = int(torch.empty(1).uniform_(n_total / 100, n_total / 2).item())
    context_mask = img.new_empty(img.size(2), img.size(3)).bernoulli_(p=num_context / n_total).bool()
    target_mask = ~context_mask
    context_mask = context_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    target_mask = target_mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return context_mask, target_mask

def np_input_to_img(x, y, img_size):
    """Given an x and y returned by a Neural Process, reconstruct image.
    Missing pixels will have a value of 0.

    Parameters
    ----------
    x : torch.Tensor
        Shape (batch_size, num_points, 2) containing normalized indices.

    y : torch.Tensor
        Shape (batch_size, num_points, num_channels) where num_channels = 1 for
        grayscale and 3 for RGB, containing normalized pixel intensities.

    img_size : tuple of ints
        [B, C, H, W].
    Returns:
        a image with size
    """
    batch_size, channel, height, width = img_size
    # Unnormalize x and y
    x = x * float(height / 2) + float(height / 2)
    x = x.long()
    # Permute y so it matches order expected by image
    # (batch_size, num_points, num_channels) -> (batch_size, num_channels, num_points)
    y = y.permute(0, 2, 1)
    # Initialize empty image
    img = torch.zeros(img_size)
    for i in range(batch_size):
        img[i, :, x[i, :, 0], x[i, :, 1]] = y[i, :, :]
    return img
