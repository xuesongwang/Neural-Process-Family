import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.basic_module import ConvDeepSet, FinalLayer, UNet
from module.utils import init_sequential_weights, init_layer_weights, to_multiple


class ConvCNP(nn.Module):
    """One-dimensional ConvCNP module.

    Args:
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, rho, points_per_unit, device = torch.device("cpu")):
        super(ConvCNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.multiplier = 2 ** self.rho.num_halving_layers
        self.device = device

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit

        # Instantiate encoder
        self.encoder = ConvDeepSet(out_channels=self.rho.in_channels,
                                   init_length_scale=init_length_scale,
                                   device = device)

        # Instantiate mean and standard deviation layers
        self.mean_layer = FinalLayer(in_channels=self.rho.out_channels,
                                     init_length_scale=init_length_scale)
        self.sigma_layer = FinalLayer(in_channels=self.rho.out_channels,
                                      init_length_scale=init_length_scale)

    def forward(self, x, y, x_out):
        """Run the module forward.

        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).

        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(x).cpu().numpy(),
                    torch.min(x_out).cpu().numpy(), -2.) - 0.1
        x_max = max(torch.max(x).cpu().numpy(),
                    torch.max(x_out).cpu().numpy(), 2.) + 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        x_grid = torch.linspace(x_min, x_max, num_points).to(self.device)
        x_grid = x_grid[None, :, None].repeat(x.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.encoder(x, y, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        h = self.rho(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # Check that shape is still fine!
        if h.shape[1] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations.
        mean = self.mean_layer(x_grid, h, x_out)
        sigma = self.sigma_fn(self.sigma_layer(x_grid, h, x_out))

        return mean, sigma

    @property
    def num_params(self):
        """Number of parameters in module."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
