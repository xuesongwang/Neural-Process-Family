import torch
import torch.nn as nn
import numpy as np
from module.basic_module import init_sequential_weights


class NeuralNetwork(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 output_dim):
        super(NeuralNetwork, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        pre_pooling_fn = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 2*self.output_dim)
        )
        self.decoder = init_sequential_weights(pre_pooling_fn)

    def forward(self, x_context):
        """Forward pass through CNP.

        Args:
            x_context (tensor): Context locations of shape
                `(batch, num_context, input_dim_x)`.
            y_context (tensor): Context values of shape
                `(batch, num_context, input_dim_y)`.
            x_target (tensor): Target locations of shape
                `(batch, num_target, input_dim_x)`.

        Returns:
            tensor: Result of forward pass.
        """
        y = self.decoder(x_context)
        mean, std = torch.split(y,self.output_dim,dim=-1)
        std = nn.functional.softplus(std)
        return mean, std

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])