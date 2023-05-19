import torch.nn as nn
import torch
import numpy as np
from module.basic_module import DeterministicDecoder, DeterministicEncoder, LatentEncoder

class NeuralProcess(nn.Module):
    """(Attentive) Neural Process module.

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
        super(NeuralProcess, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_attention = use_attention

        self.deter_encoder = \
            DeterministicEncoder(input_dim=self.input_dim + self.output_dim,
                                 attent_input_dim = self.input_dim,
                            latent_dim=self.latent_dim,
                            use_attention=use_attention)

        self.latent_encoder = \
            LatentEncoder(input_dim=self.latent_dim,
                            latent_dim=self.latent_dim)
        self.decoder = DeterministicDecoder(input_dim=self.input_dim,
                                       latent_dim=self.latent_dim,
                                       output_dim=self.output_dim,
                                       include_latent=self.use_attention)

    def forward(self, x_context, y_context, x_target, y_target = None, num_samples=20):
        """Forward pass through NP.

        Args:
            x_context (tensor): Context locations of shape
                `(batch, num_context, input_dim_x)`.
            y_context (tensor): Context values of shape
                `(batch, num_context, input_dim_y)`.
            x_target (tensor): Target locations of shape
                `(batch, num_target, input_dim_x)`.
            y_target (tensor): Target values of shape
                `(batch, num_target, input_dim_y)`.
            num_samples (int): number of z values to sample from the posterior distribution

        Returns:
            (Mean, var), dist : Result of decoder, and the distribution of the latent encoder
        """
        n = x_context.shape[1]

        r = self.deter_encoder(x_context, y_context, x_target)
        z_dist = self.latent_encoder(r)
        if y_target is None:
            z = z_dist.rsample([num_samples])
            z_dist_post = None
        else:
            r_t = self.deter_encoder(x_target, y_target, x_context)
            z_dist_post = self.latent_encoder(r_t)
            z = z_dist_post.rsample([num_samples])

        # prepare embedding concatenated with x_target
        num_target = x_target.shape[1]
        if z.shape[2] == 1:
            z = z.repeat(1, 1, num_target, 1)
        if self.use_attention == False:
            representation = z
        else:
            if r.shape[1] == 1:
                r = r.repeat(1, num_target, 1)
            r = r.unsqueeze(0).repeat(num_samples, 1, 1, 1) # n_samples
            representation = torch.cat([r, z], dim=-1)
        return self.decoder(x_target, representation, n), z_dist, z_dist_post

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])