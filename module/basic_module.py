import torch.nn as nn
import torch
import numpy as np
from module.utils import init_sequential_weights, init_layer_weights, compute_dists, pad_concat
import torch.nn.functional as F
from torch.distributions import Normal

class DotProdAttention(nn.Module):
    """
    Simple dot-product attention module. Can be used multiple times for
    multi-head attention.

    Args:
        latent_dim (int): Dimensionality of embedding for keys and queries.
        values_dim (int): Dimensionality of embedding for values.
        linear_transform (bool, optional): Use a linear for all embeddings
            before operation. Defaults to `False`.
    """

    def __init__(self, latent_dim, values_dim, linear_transform=False):
        super(DotProdAttention, self).__init__()

        self.latent_dim = latent_dim
        self.values_dim = values_dim
        self.linear_transform = linear_transform

        if self.linear_transform:
            self.key_transform = nn.Linear(self.latent_dim,
                                           self.latent_dim, bias=False)
            self.query_transform = nn.Linear(self.latent_dim,
                                             self.latent_dim, bias=False)
            self.value_transform = nn.Linear(self.values_dim,
                                             self.values_dim, bias=False)

    def forward(self, keys, queries, values):
        """Forward pass to implement dot-product attention. Assumes that
        everything is in batch mode.

        Args:
            keys (tensor): Keys of shape
                `(num_functions, num_keys, dim_key)`.
            queries (tensor): Queries of shape
                `(num_functions, num_queries, dim_query)`.
            values (tensor): Values of shape
                `(num_functions, num_values, dim_value)`.

        Returns:
            tensor: Output of shape `(num_functions, num_queries, dim_value)`.
        """
        if self.linear_transform:
            keys = self.key_transform(keys)
            queries = self.query_transform(queries)
            values = self.value_transform(values)

        dk = keys.shape[-1]
        attn_logits = torch.bmm(queries, keys.permute(0, 2, 1)) / np.sqrt(dk)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        return torch.bmm(attn_weights, values)

class MultiHeadAttention(nn.Module):
    """Implementation of multi-head attention in a batch way. Wraps around the
    dot-product attention module.

    Args:
        latent_dim (int): Dimensionality of embedding for keys, values,
            queries.
        value_dim (int): Dimensionality of values representation. Is same as
            above.
        num_heads (int): Number of dot-product attention heads in module.
    """

    def __init__(self,
                 latent_dim,
                 value_dim,
                 num_heads):
        super(MultiHeadAttention, self).__init__()

        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.value_dim = value_dim
        self.head_size = self.latent_dim // self.num_heads

        self.key_transform = nn.Linear(self.latent_dim, self.latent_dim,
                                         bias=False)
        self.query_transform = nn.Linear(self.latent_dim,
                                           self.latent_dim, bias=False)
        self.value_transform = nn.Linear(self.value_dim,
                                           self.latent_dim, bias=False)
        self.attention = DotProdAttention(latent_dim=self.latent_dim,
                                          values_dim=self.latent_dim,
                                          linear_transform=False)
        self.head_combine = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, keys, queries, values):
        """Forward pass through multi-head attention module.

        Args:
            keys (tensor): Keys of shape
                `(num_functions, num_keys, dim_key)`.
            queries (tensor): Queries of shape
                `(num_functions, num_queries, dim_query)`.
            values (tensor): Values of shape
                `(num_functions, num_values, dim_value)`.

        Returns:
            tensor: Output of shape `(num_functions, num_queries, dim_value)`.
        """
        keys = self.key_transform(keys)
        queries = self.query_transform(queries)
        values = self.value_transform(values)

        # Reshape keys, queries, values into shape
        #     (batch_size * n_heads, num_points, head_size).
        keys = self._reshape_objects(keys)
        queries = self._reshape_objects(queries)
        values = self._reshape_objects(values)

        # Compute attention mechanism, reshape, process, and return.
        attn = self.attention(keys, queries, values)
        attn = self._concat_head_outputs(attn)
        return self.head_combine(attn)

    def _reshape_objects(self, o):
        num_functions = o.shape[0]
        o = o.view(num_functions, -1, self.num_heads, self.head_size)
        o = o.permute(2, 0, 1, 3).contiguous()
        return o.view(num_functions * self.num_heads, -1, self.head_size)

    def _concat_head_outputs(self, attn):
        num_functions = attn.shape[0] // self.num_heads
        attn = attn.view(self.num_heads, num_functions, -1, self.head_size)
        attn = attn.permute(1, 2, 0, 3).contiguous()
        return attn.view(num_functions, -1, self.num_heads * self.head_size)

class CrossAttention(nn.Module):
    """Module for transformer-style cross attention to be used by the AttnCNP.

    Args:
        input_dim (int, optional): Dimensionality of the input locations.
            Defaults to `1`.
        latent_dim (int, optional): Dimensionality of the embeddings (keys).
            Defaults to `128`.
        values_dim (int, optional): Dimensionality of the embeddings (values).
            Defaults to `128`.
        num_heads (int, optional): Number of attention heads to use. Defaults
            to `8`.
    """

    def __init__(self,
                 input_dim=1,
                 latent_dim=128,
                 values_dim=128,
                 num_heads=8):
        super(CrossAttention, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.values_dim = values_dim
        self.num_heads = num_heads

        self._attention = MultiHeadAttention(latent_dim=self.latent_dim,
                                             value_dim=self.values_dim,
                                             num_heads=self.num_heads)
        self.embedding = nn.Sequential(nn.Linear(self.input_dim,  self.latent_dim),
                                        nn.ReLU())

        # Additional modules for transformer-style computations:
        self.ln1 = nn.LayerNorm(self.latent_dim)
        self.ln2 = nn.LayerNorm(self.latent_dim)
        self.ff = nn.Linear(self.latent_dim, self.latent_dim)

    def forward(self, h, x_context, x_target):
        """Forward pass through the cross-attentional mechanism.

        Args:
            h (tensor): Embeddings for context points of shape
                `(batch, num_context, latent_dim)`.
            x_context (tensor): Context locations of shape
                `(batch, num_context, input_dim)`.
            x_target (tensor): Target locations of shape
                `(batch, num_target, input_dim)`.

        Returns:
            tensor: Result of forward pass.
        """
        keys = self.embedding(x_context)
        queries = self.embedding(x_target)
        attn = self._attention(keys, queries, h)
        out = self.ln1(attn + queries)
        return self.ln2(out + self.ff(out))

class MeanPooling(nn.Module):
    """Helper class for performing mean pooling in CNPs.

    Args:
        pooling_dim (int, optional): Dimension to pool over. Defaults to `0`.
    """

    def __init__(self, pooling_dim=0):
        super(MeanPooling, self).__init__()
        self.pooling_dim = pooling_dim

    def forward(self, h, x_context=None, x_target=None):
        """Perform pooling operation.

        Args:
            h (tensor): Tensor to pool over.
            x_context (tensor): Context locations. This is not used but to consist with AttnNP
            x_target (tensor): Target locations. This is not used.
        """
        return torch.mean(h, dim=self.pooling_dim, keepdim=True)

class LatentEncoder(nn.Module):
    """Latent Encoder used for standard NP module.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        use_attention (bool, optional): Use attention. Defaults to `False`.
    """

    def __init__(self,
                 input_dim,
                 latent_dim,
                 use_lstm = False):
        super(LatentEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.use_lstm = use_lstm
        # MLP encoding
        if use_lstm == False:
            pre_pooling_fn = nn.Sequential(
                nn.Linear(self.input_dim, self.latent_dim),
                nn.ReLU(),
                # nn.Linear(self.latent_dim, self.latent_dim),
                # nn.ReLU(),
                # nn.Linear(self.latent_dim, self.latent_dim),
                # nn.ReLU(),
                nn.Linear(self.latent_dim, 2 * self.latent_dim)
            )
            self.pre_pooling_fn = init_sequential_weights(pre_pooling_fn)
        else:
        # sequential encoding
            self.pre_pooling_fn = nn.LSTM(self.input_dim, 2*self.latent_dim, batch_first=True)
        self.pooling_fn = MeanPooling(pooling_dim=1)
        self.sigma_fn = torch.sigmoid

    def forward(self, z_deter):
        """Forward pass through the decoder.

        Args:
            z_deter (tensor): the representation of context/target set passed by the determinstic part
                `(batch, num_context, latent_dim)`.

        Returns:
            tensor: Latent representation of each context set of shape
                `(batch, 1, latent_dim)`.
        """
        assert len(z_deter.shape) == 3, \
            'Incorrect shapes: ensure the context is a rank-3 tensor.'

        h = self.pre_pooling_fn(z_deter)
        if self.use_lstm == True:
            h = h[0]
        h = self.pooling_fn(h)
        mean = h[..., :self.latent_dim]
        sigma = 0.1 + 0.9* self.sigma_fn(h[..., self.latent_dim:])
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
                 use_attention=False,
                 use_lstm = False):
        super(DeterministicEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.use_attention = use_attention
        self.use_lstm = use_lstm
        # MLP encoding
        if use_lstm == False:
            pre_pooling_fn = nn.Sequential(
                nn.Linear(self.input_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim)
            )
            self.pre_pooling_fn = init_sequential_weights(pre_pooling_fn)
        else:
            # sequential encoding
            self.pre_pooling_fn = nn.LSTM(self.input_dim, self.latent_dim, batch_first=True)
        if self.use_attention:
            self.pooling_fn = CrossAttention(attent_input_dim, latent_dim=self.latent_dim, values_dim=self.latent_dim)
        else:
            self.pooling_fn = MeanPooling(pooling_dim=1)

    def forward(self, x_context, y_context, x_target=None):
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

        decoder_input = torch.cat((x_context, y_context), dim=-1)
        h = self.pre_pooling_fn(decoder_input)
        if self.use_lstm == True:
            h = h[0]
        return self.pooling_fn(h, x_context, x_target)

class DeterministicDecoder(nn.Module):
    """Decoder used for standard CNP module.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the output.
    """

    def __init__(self, input_dim, latent_dim, output_dim, include_latent=False, use_attn = False, use_lstm=False):
        super(DeterministicDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.use_attn = use_attn
        self.use_lstm = use_lstm
        # if include latent, concatenate deterministic and latent embedding before decoder
        input_latent = 2*self.latent_dim if include_latent==True else self.latent_dim
        if self.use_lstm == False:
            post_pooling_fn = nn.Sequential(
                nn.Linear(input_latent + self.input_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.ReLU(),
                nn.Linear(self.latent_dim, 2 * self.output_dim),
            )
            # use xavier initialization for faster convergence
            self.post_pooling_fn = init_sequential_weights(post_pooling_fn)
        else:
            # sequential encoding
            self.post_pooling_fn = nn.LSTM(input_latent + self.input_dim, 2 * self.output_dim, batch_first=True)
        self.sigma_fn = nn.functional.softplus

    def forward(self, x, r, n=None):
        """Forward pass through the decoder.

        Args:
            x (tensor): Target locations of shape
                `(batch, num_targets, input_dim)`.
            r (torch.tensor): Hidden representation for each task of shape
                `(batch, None, latent_dim)`.
            n (int, optional): Number of context points.

        Returns:
            tensor: Output values at each query point of shape
                `(batch, num_targets, output_dim)`
        """

        # Concatenate latents with inputs and pass through decoder.
        # Shape: (batch, num_targets, input_dim + latent_dim).
        if len(x.shape) != len(r.shape):
            x = x.unsqueeze(0).repeat(r.shape[0], 1, 1, 1) #num_samples
        z = torch.cat([x, r], -1)
        z = self.post_pooling_fn(z)
        if self.use_lstm == True:
            z = z[0]
        # Separate mean and standard deviations and return.
        mean = z[..., :self.output_dim]
        sigma = 0.1+0.9*self.sigma_fn(z[..., self.output_dim:])
        return mean, sigma

class BatchNormSequence(nn.Module):
    """Applies batch norm on features of a batch first sequence."""
    def __init__(self, out_channels, **kwargs):
        super().__init__()
        self.norm = nn.BatchNorm1d(out_channels, **kwargs)

    def forward(self, x):
        # x.shape is (Batch, Sequence, Channels)
        # Now we want to apply batchnorm and dropout to the channels. So we put it in shape
        # (Batch, Channels, Sequence) which is what BatchNorm1d expects
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        return x.permute(0, 2, 1)



class SimpleConv(nn.Module):
    """Small convolutional architecture from 1d experiments in the paper.
    This is a 4-layer convolutional network with fixed stride and channels,
    using ReLU activations.

    Args:
        in_channels (int, optional): Number of channels on the input to the
            network. Defaults to 8.
        out_channels (int, optional): Number of channels on the output by the
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8, out_channels=8):
        super(SimpleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = nn.ReLU()
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=self.in_channels, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=self.out_channels,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )
        init_sequential_weights(self.conv_net)
        self.num_halving_layers = 0

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        return self.conv_net(x)

class UNet(nn.Module):
    """Large convolutional architecture from 1d experiments in the paper.
    This is a 12-layer residual network with skip connections implemented by
    concatenation.

    Args:
        in_channels (int, optional): Number of channels on the input to
            network. Defaults to 8.
    """

    def __init__(self, in_channels=8):
        super(UNet, self).__init__()
        self.activation = nn.ReLU()
        self.in_channels = in_channels
        self.out_channels = 16
        self.num_halving_layers = 6

        self.l1 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l2 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=2 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=2 * self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        return pad_concat(x, h12)


class ConvDeepSet(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, out_channels, init_length_scale, device = torch.device("cpu")):
        super(ConvDeepSet, self).__init__()
        self.out_channels = out_channels
        self.in_channels = 2
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) *
                                  torch.ones(self.in_channels), requires_grad=True)
        self.sigma_fn = torch.exp
        self.device = device

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between x and t.

        Returns:
            tensor: Evaluation of psi(x, t) with psi an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Compute the extra density channel.
        # Shape: (batch, n_in, 1).
        density = torch.ones(batch_size, n_in, 1).to(self.device)

        # Concatenate the channel.
        # Shape: (batch, n_in, in_channels + 1).
        y_out = torch.cat([density, y], dim=2)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels + 1).
        y_out = y_out.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels + 1).
        y_out = y_out.sum(1)

        # Use density channel to normalize convolution.
        density, conv = y_out[..., :1], y_out[..., 1:]
        normalized_conv = conv / (density + 1e-8)
        y_out = torch.cat((density, normalized_conv), dim=-1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out


class FinalLayer(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_length_scale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels, init_length_scale):
        super(FinalLayer, self).__init__()
        self.out_channels = 1
        self.in_channels = in_channels
        self.g = self.build_weight_model()
        self.sigma = nn.Parameter(np.log(init_length_scale) * torch.ones(self.in_channels), requires_grad=True)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a function point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        model = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
        )
        init_sequential_weights(model)
        return model

    def rbf(self, dists):
        """Compute the RBF values for the distances using the correct length
        scales.

        Args:
            dists (tensor): Pair-wise distances between x and t.

        Returns:
            tensor: Evaluation of psi(x, t) with psi an RBF kernel.
        """
        # Compute the RBF kernel, broadcasting appropriately.
        scales = self.sigma_fn(self.sigma)[None, None, None, :]
        a, b, c = dists.shape
        return torch.exp(-0.5 * dists.view(a, b, c, -1) / scales ** 2)

    def forward(self, x, y, t):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """
        # Compute shapes.
        batch_size = x.shape[0]
        n_in = x.shape[1]
        n_out = t.shape[1]

        # Compute the pairwise distances.
        # Shape: (batch, n_in, n_out).
        dists = compute_dists(x, t)

        # Compute the weights.
        # Shape: (batch, n_in, n_out, in_channels).
        wt = self.rbf(dists)

        # Perform the weighting.
        # Shape: (batch, n_in, n_out, in_channels).
        y_out = y.view(batch_size, n_in, -1, self.in_channels) * wt

        # Sum over the inputs.
        # Shape: (batch, n_out, in_channels).
        y_out = y_out.sum(1)

        # Apply the point-wise function.
        # Shape: (batch, n_out, out_channels).
        y_out = y_out.view(batch_size * n_out, self.in_channels)
        y_out = self.g(y_out)
        y_out = y_out.view(batch_size, n_out, self.out_channels)

        return y_out