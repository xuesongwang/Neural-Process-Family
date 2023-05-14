import torch.nn as nn
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import torch
import torch.nn.functional as F
import pandas as pd
from data.GP_data_sampler import NPRegressionDescription
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import matplotlib.pyplot as plt

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
    loss = log_prob.view(*log_prob.shape[:2], -1).mean(-1)
    loss = - torch.mean(loss)
    return loss

def compute_MSE(mean, y_target):
    criterion = nn.MSELoss()
    mean_mse = criterion(mean, y_target)
    return mean_mse

def comput_kl_loss(prior, poster):
    if type(prior) == list:
        assert len(prior) == len(poster), "length of KL distributions needs to be the same"
        div = [torch.mean(kl_divergence(poster[t], prior[t]), dim=0).sum() for t in range(len(prior))]
        div = torch.stack(div).mean()
    else:
        div = kl_divergence(poster, prior)
        div = torch.mean(div, dim=0).sum()
    return div

def temp_plot(x_context, y_context, x_all, mu_s, sigma_s, pred_mu, pred_sigma):
    plt.scatter(x_context, y_context, c='red')
    plt.plot(x_all, mu_s)
    plt.plot(x_all, pred_mu, c = 'green')
    plt.fill_between(x_all, mu_s - 1.94*sigma_s, mu_s + 1.94*sigma_s, alpha=0.2)
    plt.fill_between(x_all, pred_mu - 1.94*pred_sigma, pred_mu + 1.94*pred_sigma, alpha = 0.2, color= 'green')
    plt.show()

def compute_mse_loss(x_context, y_context, x_all, predict_mean, predict_var):
    # genertate x_all
    # mu_s, sigma_s = posterior_predictive(x_all, x_context, y_context)
    criterion = nn.MSELoss()
    kernel = C(1.0, (1e-2, 1)) * RBF(0.4, (1e-2, 1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp.fit(to_numpy(x_context[0]), to_numpy(y_context[0]))
    mu_s, sigma_s = gp.predict(to_numpy(x_all[0]), return_std=True)
    # temp_plot(to_numpy(x_context[0]), to_numpy(y_context[0]), to_numpy(x_all[0,:,0]), mu_s[:,0], sigma_s,
    #           predict_mean[0,:,0].detach().cpu(), predict_var[0,:,0].detach().cpu())
    mean_mse = criterion(predict_mean[0].detach().cpu(), torch.tensor(mu_s))
    std_mse = criterion(predict_var[0,:,0].detach().cpu(), torch.tensor(sigma_s))
    return mean_mse, std_mse

def to_numpy(var):
    return var.detach().cpu().numpy()

def to_tensor(var):
    var = torch.tensor(var,dtype=torch.float32).unsqueeze(0)
    return var if len(var.shape) > 2 else var.unsqueeze(-1)

def normalize(y, mean = None, std = None):
    if mean is None:
        mean = torch.mean(y,dim=[1,2])[:, None, None]
        std = torch.std(y, dim=[1,2])[:, None, None]
    y = (y - mean)/std
    return y, mean, std

def save_plot_data(data_test, kernel, sequential = False):
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
        if sequential:
            data = data_test.generate_temporal_curves(include_context=False)
        else:
            data = data_test.generate_curves(include_context=False)
        (x_context, y_context), x_target = data.query
        y_target = data.y_target
    context = to_numpy(torch.cat([x_context[0], y_context[0]], dim=-1))
    # print ("context data shape:", context.shape)
    df = pd.DataFrame(context)
    if sequential:
        df.to_csv('saved_fig/csv/plot_sequential_context_' + kernel + '.csv', index=False)
    else:
        df.to_csv('saved_fig/csv/plot_context_'+kernel+'.csv', index=False)
    target = to_numpy(torch.cat([x_target[0],y_target[0]], dim=-1))
    # print("target data shape:", target.shape)
    df = pd.DataFrame(target)
    if sequential:
        df.to_csv('saved_fig/csv/plot_sequential_target_' + kernel + '.csv', index=False)
    else:
        df.to_csv('saved_fig/csv/plot_target_' + kernel + '.csv', index=False)
    return data

def load_plot_data(kernel, sequential = False):
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
        if sequential:
            context = pd.read_csv('saved_fig/csv/plot_sequential_context_' + kernel + '.csv')
            target = pd.read_csv('saved_fig/csv/plot_sequential_target_' + kernel + '.csv')
        else:
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

def channels_to_2nd_dim(X):
    """
    Takes a signal with channels on the last dimension (for most operations) and
    returns it with channels on the second dimension (for convolutions).
    """
    return X.permute(*([0, X.dim() - 1] + list(range(1, X.dim() - 1))))

def channels_to_last_dim(X):
    """
    Takes a signal with channels on the second dimension (for convolutions) and
    returns it with channels on the last dimension (for most operations).
    """
    return X.permute(*([0] + list(range(2, X.dim())) + [1]))

def make_abs_conv(Conv):
    """Make a convolution have only positive parameters."""

    class AbsConv(Conv):
        def forward(self, input):
            return F.conv2d(input,self.weight.abs(),self.bias,self.stride,self.padding,
                self.dilation,self.groups,)
    return AbsConv

def make_depth_sep_conv(Conv):
    """Make a convolution module depth separable."""

    class DepthSepConv(nn.Module):
        """Make a convolution depth separable.
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
        **kwargs :
            Additional arguments to `Conv`
        """

        def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            confidence=False,
            bias=True,
            **kwargs
        ):
            super().__init__()
            self.depthwise = Conv(
                in_channels,
                in_channels,
                kernel_size,
                groups=in_channels,
                bias=bias,
                **kwargs
            )
            self.pointwise = Conv(in_channels, out_channels, 1, bias=bias)
            self.reset_parameters()

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out

        def reset_parameters(self):
            weights_init(self)

    return DepthSepConv




def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(module, activation="relu"):
    """Initialize a linear layer.
    Parameters
    ----------
    module : nn.Module
       module to initialize.
    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight

    if module.bias is not None:
        module.bias.data.zero_()

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity="relu")
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))

def weights_init(module, **kwargs):
    """Initialize a module and all its descendents.
    Parameters
    ----------
    module : nn.Module
       module to initialize.
    """
    module.is_resetted = True
    for m in module.modules():
        try:
            if hasattr(module, "reset_parameters") and module.is_resetted:
                # don't reset if resetted already (might want special)
                continue
        except AttributeError:
            pass

        if isinstance(m, torch.nn.modules.conv._ConvNd):
            # used in https://github.com/brain-research/realistic-ssl-evaluation/
            nn.init.kaiming_normal_(m.weight, mode="fan_out", **kwargs)
        elif isinstance(m, nn.Linear):
            linear_init(m, **kwargs)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

def init_param_(param, activation=None, is_positive=False, bound=0.05, shift=0):
    """Initialize inplace some parameters of the model that are not part of a
    children module.
    Parameters
    ----------
    param : nn.Parameter:
        Parameters to initialize.
    activation : torch.nn.modules.activation or str, optional)
        Activation that will be used on the `param`.
    is_positive : bool, optional
        Whether to initilize only with positive values.
    bound : float, optional
        Maximum absolute value of the initealized values. By default `0.05` which
        is keras default uniform bound.
    shift : int, optional
        Shift the initialisation by a certain value (same as adding a value after init).
    """
    gain = get_gain(activation)
    if is_positive:
        nn.init.uniform_(param, 1e-5 + shift, bound * gain + shift)
        return

    nn.init.uniform_(param, -bound * gain + shift, bound * gain + shift)

def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape (batch, n, 1).
        y (tensor): Inputs of shape (batch, m, 1).

    Returns:
        tensor: Pair-wise distances of shape (batch, n, m).
    """
    return (x - y.permute(0, 2, 1)) ** 2

def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple

def pad_concat(t1, t2):
    """Concat the activations of two layer channel-wise by padding the layer
    with fewer points with zeros.

    Args:
        t1 (tensor): Activations from first layers of shape `(batch, n1, c1)`.
        t2 (tensor): Activations from second layers of shape `(batch, n2, c2)`.

    Returns:
        tensor: Concatenated activations of both layers of shape
            `(batch, max(n1, n2), c1 + c2)`.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:  # Even difference
            t2 = F.pad(t2, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t2 = F.pad(t2, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:  # Even difference
            t1 = F.pad(t1, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t1 = F.pad(t1, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')

    return torch.cat([t1, t2], dim=1)
