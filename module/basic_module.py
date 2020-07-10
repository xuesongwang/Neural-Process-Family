import torch.nn as nn
import torch
import numpy as np
from module.utils import init_sequential_weights
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
        self.value_transform = nn.Linear(self.latent_dim,
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

    def forward(self, h, x_context, x_target):
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
            nn.ReLU(),
            nn.Linear(self.latent_dim, 2 * self.latent_dim)
        )
        self.pre_pooling_fn = init_sequential_weights(pre_pooling_fn)
        self.pooling_fn = MeanPooling(pooling_dim=1)
        self.sigma_fn = torch.sigmoid

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
        h = self.pooling_fn(h, x_context, x_target)
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
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.pre_pooling_fn = init_sequential_weights(pre_pooling_fn)
        if self.use_attention:
            self.pooling_fn = CrossAttention(attent_input_dim)
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
        return self.pooling_fn(h, x_context, x_target)

class DeterministicDecoder(nn.Module):
    """Decoder used for standard CNP module.

    Args:
        input_dim (int): Dimensionality of the input.
        latent_dim (int): Dimensionality of the hidden representation.
        output_dim (int): Dimensionality of the output.
    """

    def __init__(self, input_dim, latent_dim, output_dim, include_latent=False):
        super(DeterministicDecoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        # if include latent, concatenate deterministic and latent embedding before decoder
        input_latent = 2*self.latent_dim if include_latent==True else self.latent_dim
        post_pooling_fn = nn.Sequential(
            nn.Linear(input_latent + self.input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, 2 * self.output_dim),
        )
        # use xavier initialization for faster convergence
        self.post_pooling_fn = init_sequential_weights(post_pooling_fn)
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
        z = torch.cat([x, r], -1)
        z = self.post_pooling_fn(z)

        # Separate mean and standard deviations and return.
        mean = z[..., :self.output_dim]
        sigma = self.sigma_fn(z[..., self.output_dim:])
        return mean, sigma


