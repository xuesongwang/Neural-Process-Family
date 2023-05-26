import copy

import numpy as np
import torch
import torch.nn as nn
from module.basic_module import ConvDeepSet, FinalLayer, UNet, SimpleConv
from module.basic_module import LatentEncoder
from module.utils import init_sequential_weights, init_layer_weights, to_multiple



class ConvNP(nn.Module):
    """One-dimensional ConvCNP module.

    Args:
        learn_length_scale (bool): Learn the length scale.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    def __init__(self, rho, points_per_unit, device = torch.device("cpu")):
        super(ConvNP, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()
        self.rho = rho
        self.multiplier = 2 ** self.rho.num_halving_layers
        self.device = device

        # Compute initialisation.
        self.points_per_unit = points_per_unit
        init_length_scale = 2.0 / self.points_per_unit

        # Instantiate encoder
        self.deter_encoder = ConvDeepSet(out_channels=self.rho.in_channels,
                                   init_length_scale=init_length_scale,
                                   device = device)
        self.latent_encoder = LatentEncoder(input_dim=2*self.rho.in_channels,
                          latent_dim=self.rho.in_channels, use_pooling=False) # 2* because of the residual concatenation of UNet

        self.decoder = copy.copy(rho) #anothother CNN

        # Instantiate mean and standard deviation layers
        self.mean_layer = FinalLayer(in_channels=self.rho.in_channels*2,
                                     init_length_scale=init_length_scale)
        self.sigma_layer = FinalLayer(in_channels=self.rho.in_channels*2,
                                      init_length_scale=init_length_scale)

    def forward(self, x_context, y_context, x_target, y_target = None, num_samples=20):
        """Run the module forward.

        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).

        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        # Determine the grid on which to evaluate functional representation.
        x_min = min(torch.min(x_context).cpu().numpy(),
                    torch.min(x_target).cpu().numpy(), -2.) - 0.1
        x_max = max(torch.max(x_context).cpu().numpy(),
                    torch.max(x_target).cpu().numpy(), 2.) + 0.1
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),
                                     self.multiplier))
        x_grid = torch.linspace(x_min, x_max, num_points).to(self.device)
        x_grid = x_grid[None, :, None].repeat(x_context.shape[0], 1, 1)

        # Apply first layer and conv net. Take care to put the axis ranging
        # over the data last.
        h = self.activation(self.deter_encoder(x_context, y_context, x_grid))
        h = h.permute(0, 2, 1)
        h = h.reshape(h.shape[0], h.shape[1], num_points)
        h = self.rho(h)
        h = h.reshape(h.shape[0], h.shape[1], -1).permute(0, 2, 1)

        # TODO: split and formulate z, then pass it to CNN
        z_dist = self.latent_encoder(h)
        if y_target is None:
            z = z_dist.rsample([num_samples])
            z_dist_post = None
        else:
            h_t = self.activation(self.deter_encoder(x_target, y_target, x_grid))
            h_t = h_t.permute(0, 2, 1)
            h_t = h_t.reshape(h_t.shape[0], h_t.shape[1], num_points)
            h_t = self.rho(h_t)
            h_t = h_t.reshape(h_t.shape[0], h_t.shape[1], -1).permute(0, 2, 1)
            z_dist_post = self.latent_encoder(h_t)
            z = z_dist_post.rsample([num_samples])

        z = z.permute(0, 1, 3, 2)  #(n_samples, bs, z_dim, n_points)
        batch_size = z.shape[1]
        z = z.reshape(num_samples*batch_size, z.shape[2], num_points)
        z = self.decoder(z) # CNN module
        z = z.permute(0, 2, 1)
        z = z.reshape(num_samples, batch_size, num_points, -1)

        # Check that shape is still fine!
        if z.shape[2] != x_grid.shape[1]:
            raise RuntimeError('Shape changed.')

        # Produce means and standard deviations.
        mean = self.mean_layer(x_grid, z, x_target)
        sigma = 0.1 + 0.9 * self.sigma_fn(self.sigma_layer(x_grid, z, x_target))
        return (mean, sigma), z_dist, z_dist_post

    @property
    def num_params(self):
        """Number of parameters in module."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
