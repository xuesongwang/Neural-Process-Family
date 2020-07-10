import torch
import torch.nn as nn
import numpy as np
from module.basic_module import DeterministicEncoder, DeterministicDecoder

class ConditionalNeuralProcess(nn.Module):
    """Conditional Neural Process module.

    See https://arxiv.org/abs/1807.01613 for details.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the input signal.
        use_attention (bool, optional): Switch between ANPs and CNPs. Defaults
            to `False`.
    """

    def __init__(self,
                 input_dim,
                 latent_dim,
                 output_dim,
                 use_attention=False):
        super(ConditionalNeuralProcess, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_attention = use_attention

        self.encoder = \
            DeterministicEncoder(input_dim=self.input_dim + self.output_dim,
                            latent_dim=self.latent_dim,
                            use_attention=use_attention)
        self.decoder = DeterministicDecoder(input_dim=self.input_dim,
                                       latent_dim=self.latent_dim,
                                       output_dim=self.output_dim)

    def forward(self, x_context, y_context, x_target):
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
        n = x_context.shape[1]
        r = self.encoder(x_context, y_context)
        # If latent representation is global, repeat once for each input.
        if r.shape[1] == 1:
            r = r.repeat(1, x_target.shape[1], 1)
        return self.decoder(x_target, r, n)

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])