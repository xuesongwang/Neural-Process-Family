import torch.nn as nn
import torch
import numpy as np
from module.basic_module import DeterministicDecoder, MeanPooling, CrossAttention
from module.utils import init_sequential_weights
from torch.distributions import Normal


class LatentEncoder(nn.Module):
    """Latent Encoder used for standard SNP module.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        use_attention (bool, optional): Use attention. Defaults to `False`.
    """

    def __init__(self,
                 input_dim,
                 latent_dim):
        super(LatentEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        pre_pooling_fn = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU()
        )
        final_fn = nn.Sequential(nn.ReLU(), nn.Linear(self.latent_dim, 2 * self.latent_dim))
        self.pre_pooling_fn = init_sequential_weights(pre_pooling_fn)
        self.final_fn = init_sequential_weights(final_fn)
        self.sigma_fn = torch.sigmoid

    def forward(self, x_context, y_context, latent_rnn_hidden=None, get_hidden=False, given_hidden=None):
        """Forward pass through the encoder.

        Args:
            x_context (tensor): Context locations of shape
                `(batch, num_context, input_dim_x)`.
            y_context (tensor): Context values of shape
                `(batch, num_context, input_dim_y)`.
            x_target (tensor, optional): Target locations of shape
                `(batch, num_target, input_dim_x)`.
            latent_rnn_hidden: sequential part of the latent representation, if not None, the shape should be `(batch, latent_dim)`
            given_hidden: True/False, whether non-sequential part of latent hidden layers are given
            get_hidden :  non-sequential part of the latent representation, if not None, the shape should be `(batch, latent dim)`

        Returns:
            tensor: Latent representation of each context set of shape
                `(batch, 1, latent_dim)`.
        """
        assert len(x_context.shape) == 3, \
            'Incorrect shapes: ensure x_context is a rank-3 tensor.'
        assert len(y_context.shape) == 3, \
            'Incorrect shapes: ensure y_context is a rank-3 tensor.'
        if given_hidden is None:
            encoder_input = torch.cat((x_context, y_context), dim=-1)
            h = self.pre_pooling_fn(encoder_input)
            if get_hidden:
                return h
        else:
            h = given_hidden

        h = torch.sum(h, dim=1)

        # add sequential hidden layer
        if latent_rnn_hidden is not None:
            h += latent_rnn_hidden

        h = self.final_fn(h)
        mean = h[..., :self.latent_dim]
        sigma = self.sigma_fn(h[..., self.latent_dim:])
        dist = Normal(loc=mean, scale=sigma)
        return dist

class DeterministicEncoder(nn.Module):
    """Encoder used for standard CNP module.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        use_attention (bool, optional): Use attention. Defaults to `False`.
    """

    def __init__(self,
                 input_dim,
                 latent_dim,
                 attent_input_dim = 1,
                 use_attention=False):
        super(DeterministicEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.use_attention = use_attention

        pre_pooling_fn = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.pre_pooling_fn = init_sequential_weights(pre_pooling_fn)
        if self.use_attention:
            self.pooling_fn = CrossAttention(attent_input_dim)
        else:
            self.pooling_fn = MeanPooling(pooling_dim=1)

    def forward(self, x_context, y_context, x_target=None, deter_rnn_hidden=None, get_hidden=False, given_hidden=None):
        """Forward pass through the decoder.

        Args:
            x_context (tensor): Context locations of shape
                `(batch, num_context, input_dim_x)`.
            y_context (tensor): Context values of shape
                `(batch, num_context, input_dim_y)`.
            x_target (tensor, optional): Target locations of shape
                `(batch, num_target, input_dim_x)`.

        Returns:
            tensor: Latent representation of each context set of shape
                `(batch, 1, latent_dim)`.
        """
        assert len(x_context.shape) == 3, \
            'Incorrect shapes: ensure x_context is a rank-3 tensor.'
        assert len(y_context.shape) == 3, \
            'Incorrect shapes: ensure y_context is a rank-3 tensor.'
        if given_hidden is None:
            encoder_input = torch.cat((x_context, y_context), dim=-1)
            h = self.pre_pooling_fn(encoder_input)
            if get_hidden:
                return h
        else:
            h = given_hidden
        # apply attention or mean
        h = self.pooling_fn(h, x_context, x_target)

        # add sequential hidden layer
        if deter_rnn_hidden is not None:
            h += deter_rnn_hidden.unsqueeze(1)

        return h


class SequentialNeuralProcess(nn.Module):
    """Sequential Neural Process module.

    See https://arxiv.org/abs/1906.10264 for details.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the input signal.
        beta (float):  trade-off between posterior drop-out and
        use_attention (bool, optional): Switch between ANPs and CNPs. Defaults
            to `False`.
    """

    def __init__(self,
                 input_dim,
                 latent_dim,
                 output_dim,
                 beta = 1.0,
                 device = torch.device("cpu"),
                 use_attention=False):
        super(SequentialNeuralProcess, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.beta = beta
        self.device = device
        self.use_attention = use_attention
        self.deter_encoder = \
            DeterministicEncoder(input_dim=self.input_dim + self.output_dim,
                                 attent_input_dim = self.input_dim,
                            latent_dim=self.latent_dim,
                            use_attention=use_attention)

        self.latent_encoder = \
            LatentEncoder(input_dim=self.input_dim + self.output_dim,
                            latent_dim=self.latent_dim)
        self.decoder = DeterministicDecoder(input_dim=self.input_dim,
                                       latent_dim=self.latent_dim,
                                       output_dim=self.output_dim,
                                       include_latent=True)
        # rnn modules
        self.deter_rnn = nn.LSTMCell(input_size=self.latent_dim, hidden_size=self.latent_dim)
        self.latent_rnn = nn.LSTMCell(input_size=self.latent_dim, hidden_size=self.latent_dim)


    def forward(self, x_context, y_context, x_target, y_target = None):
        """Forward pass through NP.

        Args:
            x_context (tensor): Context locations of shape
                `(batch, num_context, input_dim_x)`.
            y_context (tensor): Context values of shape
                `(batch, num_context, input_dim_y)`.
            x_target (tensor): Target locations of shape
                `(batch, num_target, input_dim_x)`.

        Returns:
            (Mean, var), dist : Result of decoder, and the distribution of the latent encoder
        """
        len_seq = x_context.shape[1]
        batch_size = x_context.shape[0]

        # latent rnn state initialization
        latent_rnn_state = torch.zeros(batch_size, self.latent_dim).to(self.device)
        latent_rnn_hidden =  torch.zeros(batch_size, self.latent_dim).to(self.device)
        latent_rnn_input = latent_rnn_hidden

        # deterministic rnn state initialization
        deter_rnn_state = torch.zeros(batch_size, self.latent_dim).to(self.device)
        deter_rnn_hidden = torch.zeros(batch_size, self.latent_dim).to(self.device)
        deter_rnn_input = deter_rnn_hidden

        mean_list, sigma_list = [], []
        prior_list, post_list = [], []
        for t in range(len_seq):
            # LATENT ENCODER
            # (non-sequential part)
            latent_hidden = self.latent_encoder(x_context[:,[t],:], y_context[:,[t],:], get_hidden = True)
            latent_hidden_mean = torch.mean(latent_hidden, dim=1)
            latent_rnn_input = latent_rnn_input + latent_hidden_mean # can not use += to avoid inplace operation
            # update latent rnn (sequential part)
            latent_rnn_hidden, latent_rnn_state = self.latent_rnn(latent_rnn_input, (latent_rnn_hidden, latent_rnn_state))
            # incorporate updated hidden state into encoder
            prior = self.latent_encoder(x_context[:,[t],:], y_context[:,[t],:], latent_rnn_hidden,
                                                        given_hidden = latent_hidden)
            if y_target is None:
                z = prior.rsample()
                post = None
            else:
                post = self.latent_encoder(x_target[:,[t],:], y_target[:,[t],:], latent_rnn_hidden)
                z = post.rsample()
            z = z.unsqueeze(dim=1)

            # DETERMINISTIC ENCODER
            #(non-sequential part)
            deter_hidden = self.deter_encoder(x_context[:,[t],:], y_context[:,[t],:], get_hidden=True)
            deter_hidden_mean = torch.mean(deter_hidden, dim=1)
            deter_rnn_input = deter_rnn_input + deter_hidden_mean
            # update latent rnn (sequential part)
            deter_rnn_hidden, deter_rnn_state = self.deter_rnn(deter_rnn_input, (deter_rnn_hidden, deter_rnn_state))
            # incorporate updated hidden state into encoder
            r = self.deter_encoder(x_context[:,[t],:], y_context[:,[t],:], x_target[:,[t],:], deter_rnn_hidden,
                                             given_hidden=deter_hidden)

            # REPRESENTATION MERGING
            representation = torch.cat([r, z], dim=-1)

            # DECODER
            mean, sigma = self.decoder(x_target[:,[t],:], representation, 1)
            mean_list.append(mean)
            sigma_list.append(sigma)
            prior_list.append(prior)
            post_list.append(post)
        mean_list = torch.cat(mean_list,dim=1)
        sigma_list = torch.cat(sigma_list, dim=1)
        return (mean_list, sigma_list), prior_list, post_list

    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])