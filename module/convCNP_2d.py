import random
from module.utils import generate_mask, channels_to_2nd_dim, channels_to_last_dim, \
    make_abs_conv, make_depth_sep_conv, weights_init, linear_init
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

class ResConvBlock(nn.Module):
    """Convolutional block inspired by the pre-activation Resnet [1]
    and depthwise separable convolutions [2].
    Parameters
    ----------
    in_chan : int
        Number of input channels.
    out_chan : int
        Number of output channels.
    Conv : nn.Module
        Convolutional layer (unitialized). E.g. `nn.Conv1d`.
    kernel_size : int or tuple, optional
        Size of the convolving kernel. Should be odd to keep the same size.
    activation: callable, optional
        Activation object. E.g. `nn.RelU()`.
    Normalization : nn.Module, optional
        Normalization layer (unitialized). E.g. `nn.BatchNorm1d`.
    n_conv_layers : int, optional
        Number of convolutional layers, can be 1 or 2.
    is_bias : bool, optional
        Whether to use a bias.
    References
    ----------
    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016, October). Identity mappings
        in deep residual networks. In European conference on computer vision
        (pp. 630-645). Springer, Cham.
    [2] Chollet, F. (2017). Xception: Deep learning with depthwise separable
        convolutions. In Proceedings of the IEEE conference on computer vision
        and pattern recognition (pp. 1251-1258).
    """

    def __init__(
        self,
        in_chan,
        out_chan,
        Conv,
        kernel_size=5,
        activation=nn.ReLU(),
        Normalization=nn.Identity,
        is_bias=True,
        n_conv_layers=1,
    ):
        super().__init__()
        self.activation = activation
        self.n_conv_layers = n_conv_layers
        assert self.n_conv_layers in [1, 2]

        if kernel_size % 2 == 0:
            raise ValueError("`kernel_size={}`, but should be odd.".format(kernel_size))

        padding = kernel_size // 2

        if self.n_conv_layers == 2:
            self.norm1 = Normalization(in_chan)
            self.conv1 = make_depth_sep_conv(Conv)(
                in_chan, in_chan, kernel_size, padding=padding, bias=is_bias
            )
        self.norm2 = Normalization(in_chan)
        self.conv2_depthwise = Conv(
            in_chan, in_chan, kernel_size, padding=padding, groups=in_chan, bias=is_bias
        )
        self.conv2_pointwise = Conv(in_chan, out_chan, 1, bias=is_bias)

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def forward(self, X):

        if self.n_conv_layers == 2:
            out = self.conv1(self.activation(self.norm1(X)))
        else:
            out = X

        out = self.conv2_depthwise(self.activation(self.norm2(out)))
        # adds residual before point wise => output can change number of channels
        out = out + X
        out = self.conv2_pointwise(out.contiguous())  # for some reason need contiguous
        return out

class MLP(nn.Module):
    """General MLP class.

    Parameters
    ----------
    input_size: int

    output_size: int

    hidden_size: int, optional
        Number of hidden neurones.

    n_hidden_layers: int, optional
        Number of hidden layers.

    activation: callable, optional
        Activation function. E.g. `nn.RelU()`.

    is_bias: bool, optional
        Whether to use biaises in the hidden layers.

    dropout: float, optional
        Dropout rate.

    is_force_hid_smaller : bool, optional
        Whether to force the hidden dimensions to be smaller or equal than in and out.
        If not, it forces the hidden dimension to be larger or equal than in or out.

    is_res : bool, optional
        Whether to use residual connections.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=32,
        n_hidden_layers=1,
        activation=nn.ReLU(),
        is_bias=True,
        dropout=0,
        is_force_hid_smaller=False,
        is_res=False,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.is_res = is_res

        if is_force_hid_smaller and self.hidden_size > max(
            self.output_size, self.input_size
        ):
            self.hidden_size = max(self.output_size, self.input_size)
            txt = "hidden_size={} larger than output={} and input={}. Setting it to {}."
        elif self.hidden_size < min(self.output_size, self.input_size):
            self.hidden_size = min(self.output_size, self.input_size)
            txt = (
                "hidden_size={} smaller than output={} and input={}. Setting it to {}."
            )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.activation = activation

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size, bias=is_bias)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size, bias=is_bias)
                for _ in range(self.n_hidden_layers - 1)
            ]
        )
        self.out = nn.Linear(self.hidden_size, self.output_size, bias=is_bias)

        self.reset_parameters()

    def forward(self, x):
        out = self.to_hidden(x)
        out = self.activation(out)
        x = self.dropout(out)

        for linear in self.linears:
            out = linear(x)
            out = self.activation(out)
            if self.is_res:
                out = out + x
            out = self.dropout(out)
            x = out

        out = self.out(x)
        return out

    def reset_parameters(self):
        linear_init(self.to_hidden, activation=self.activation)
        for lin in self.linears:
            linear_init(lin, activation=self.activation)
        linear_init(self.out)

class CNN(nn.Module):
    """Simple multilayer CNN.
    Parameters
    ----------
    n_channels : int or list
        Number of channels, same for input and output. If list then needs to be
        of size `n_blocks - 1`, e.g. [16, 32, 64] means that you will have a
        `[ConvBlock(16,32), ConvBlock(32, 64)]`.
    ConvBlock : nn.Module
        Convolutional block (unitialized). Needs to take as input `Should be
        initialized with `ConvBlock(in_chan, out_chan)`.
    n_blocks : int, optional
        Number of convolutional blocks.
    is_chan_last : bool, optional
        Whether the channels are on the last dimension of the input.
    kwargs :
        Additional arguments to `ConvBlock`.
    """

    def __init__(self, n_channels, ConvBlock, n_blocks=3, is_chan_last=False):

        super().__init__()
        self.n_blocks = n_blocks
        self.is_chan_last = is_chan_last
        self.in_out_channels = self._get_in_out_channels(n_channels, n_blocks)
        self.conv_blocks = nn.ModuleList([ConvBlock for in_chan, out_chan in self.in_out_channels])
        self.is_return_rep = False  # never return representation for vanilla conv

        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def _get_in_out_channels(self, n_channels, n_blocks):
        """Return a list of tuple of input and output channels."""
        if isinstance(n_channels, int):
            channel_list = [n_channels] * (n_blocks + 1)
        else:
            channel_list = list(n_channels)

        assert len(channel_list) == (n_blocks + 1), "{} != {}".format(
            len(channel_list), n_blocks + 1
        )

        return list(zip(channel_list, channel_list[1:]))

    def forward(self, X):
        if self.is_chan_last:
            X = channels_to_2nd_dim(X)

        X, representation = self.apply_convs(X)

        if self.is_chan_last:
            X = channels_to_last_dim(X)

        if self.is_return_rep:
            return X, representation

        return X

    def apply_convs(self, X):
        for conv_block in self.conv_blocks:
            X = conv_block(X)
        return X, None

def channel_first(x):
    return x.permute(0,3,1,2)

class ConvCNP2d(nn.Module):
    def __init__(self, channel=1, kernel_size = 9):
        super().__init__()
        self.channel = channel
        Conv = lambda y_dim: make_abs_conv(nn.Conv2d)(
            y_dim,y_dim, groups=y_dim,kernel_size=11,padding=11 // 2,bias=False,
        )
        self.conv_theta = Conv(channel)
        self.resizer = nn.Linear(self.channel * 2, 128)
        self.decoder = MLP(input_size=128, output_size= self.channel*2, n_hidden_layers=4, hidden_size=128)
        # self.decoder = nn.Linear(128, self.channel *2)
        res_kernel_size = 3 if kernel_size!=9 else 5

        self.cnn = CNN(n_channels=128,
                       ConvBlock=ResConvBlock(in_chan=128, out_chan=128,
                                              Conv=nn.Conv2d, kernel_size=9,
                                              n_conv_layers=2),
                       n_blocks=5,
                       is_chan_last=True,
                       )

        self.pos = nn.Softplus()
        self.mr = [0.5, 0.7, 0.9]
        self.reset_parameters()

    def reset_parameters(self):
        weights_init(self)

    def context_to_induced(self, mask_cntxt, X):
        """Infer the missing values  and compute a density channel."""

        # channels have to be in second dimension for convolution
        # size = [batch_size, y_dim, *grid_shape]
        # X = channels_to_2nd_dim(X)
        # size = [batch_size, x_dim, *grid_shape]
        # mask_cntxt = channels_to_2nd_dim(mask_cntxt).float()
        mask_cntxt = mask_cntxt.unsqueeze(1).repeat(1, X.size(1), 1, 1)
        # size = [batch_size, y_dim, *grid_shape]
        X_cntxt = X * mask_cntxt
        signal = self.conv_theta(X_cntxt)
        density = self.conv_theta(mask_cntxt.float())

        # normalize
        out = signal / torch.clamp(density, min=1e-5)

        # size = [batch_size, y_dim * 2, *grid_shape]
        out = torch.cat([out, density], dim=1)

        # size = [batch_size, *grid_shape, y_dim * 2]
        out = channels_to_last_dim(out)

        # size = [batch_size, *grid_shape, r_dim]
        out = self.resizer(out)

        return out


    def forward(self, img):
        # generate mask phase
        context_mask, _ = generate_mask(img)
        h = self.context_to_induced(context_mask, img)


        # get context into induced
        # signal = I * M_c
        # density = M_c
        #
        #
        # # self.conv_theta.abs_constraint()
        # density_prime = self.conv_theta(density)
        # signal_prime = self.conv_theta(signal)
        # # signal_prime = signal_prime.div(density_prime + 1e-8)
        # # # self.conv_theta.abs_unconstraint()
        # h = torch.cat([signal_prime, density_prime], 1)


        # CNN resblock part
        f = self.cnn(h)
        y = self.decoder(f)
        # print(y.shape)
        mean, std = y.split(self.channel, dim=-1)
        std = self.pos(std)
        mean = channel_first(mean)
        std = channel_first(std)
        # print(mean.shape, std.shape)
        return mean, std

