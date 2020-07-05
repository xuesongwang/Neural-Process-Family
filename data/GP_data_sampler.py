import numpy as np
import collections
import torch

# The NP takes as input a `NPRegressionDescription` namedtuple with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tesor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that describes the number
#     of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "y_target", "num_total_points", "num_context_points"))


class GPCurvesReader(object):
    """Generates curves using a Gaussian Process (GP).

    Supports vector inputs (x) and vector outputs (y). Kernel is
    mean-squared exponential, using the x-value l2 coordinate distance scaled by
    some factor chosen randomly in a range. Outputs are independent gaussian
    processes.
    """
    def __init__(self,
                 kernel,
                 batch_size,
                 max_num_context,
                 x_size=1,
                 y_size=1,
                 testing=False,
                 device = torch.device("cpu")):
        """Creates a regression dataset of functions sampled from a GP.

        Args:
        kernel: kernel type, "EQ" or "period"
        batch_size: An integer.
        max_num_context: The max number of observations in the context.
        x_size: Integer >= 1 for length of "x values" vector.
        y_size: Integer >= 1 for length of "y values" vector.
        l1_scale: Float; typical scale for kernel distance function.
        sigma_scale: Float; typical scale for variance.
        testing: Boolean that indicates whether we are testing. If so there are
        more targets for visualization.
        """
        self.kernel = kernel
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._x_size = x_size
        self._y_size = y_size
        self._testing = testing
        self.device = device

    def _rbf_kernel(self, xdata, l1 = 0.4, sigma_f = 1.0, sigma_noise=2e-2):
        """Applies the Gaussian kernel to generate curve data
            we use the same kernel parameter for the whole training process
            instead of using dynamic parameters
        Args:
        xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
            the values of the x-axis data.
        l1: Tensor with shape `[batch_size, y_size, x_size]`, the scale
            parameter of the Gaussian kernel.
        sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
            of the std.
        sigma_noise: Float, std of the noise that we add for stability.

        Returns:
        The kernel, a float tensor with shape
        `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        # Set kernel parameters
        l1 = torch.ones([self._batch_size, self._y_size, self._x_size]).to(self.device) * l1
        sigma_f = torch.ones([self._batch_size, self._y_size]).to(self.device) * sigma_f

        num_total_points = xdata.size(1)
        # Expand and take the difference
        xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]
        diff = xdata1 - xdata2  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :])**2

        norm = torch.sum(norm,  dim=-1)  # [B, y_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = (sigma_f**2)[:, :, None, None] * torch.exp(-0.5 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise**2) * torch.eye(num_total_points).to(self.device)

        return kernel

    def _periodic_kernel(self, xdata, l1 = 1.0, p = 1.0, sigma_f = 1.0, sigma_noise=2e-2):
        """Applies the periodic kernel to generate curve data
            we use the same kernel parameter for the whole training process
            instead of using dynamic parameters
        Args:
        xdata: Tensor with shape `[batch_size, num_total_points, x_size]` with
            the values of the x-axis data.
        l1:Tensor with shape `[batch_size, y_size, x_size]`, the scale
            parameter of the Gaussian kernel.
        p:  Tensor with the shape `[batch_size, y_size, x_size]`, the distance between repetitions.
        sigma_f: Float tensor with shape `[batch_size, y_size]`; the magnitude
            of the std.
        sigma_noise: Float, std of the noise that we add for stability.

        Returns:
        The kernel, a float tensor with shape
        `[batch_size, y_size, num_total_points, num_total_points]`.
        """
        l1 = torch.ones([self._batch_size, self._y_size, self._x_size]).to(self.device) * l1
        sigma_f = torch.ones([self._batch_size, self._y_size]).to(self.device) * sigma_f

        num_total_points = xdata.size(1)
        # Expand and take the difference
        xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]

        diff = np.pi*torch.abs(xdata1 - xdata2)/p  # [B, num_total_points, num_total_points, x_size]

        # [B, y_size, num_total_points, num_total_points, x_size]
        norm = (diff[:, None, :, :, :] / l1[:, :, None, None, :]) ** 2
        norm = torch.sum(norm, dim=-1) # [B, y_size, num_total_points, num_total_points]

        # [B, y_size, num_total_points, num_total_points]
        kernel = (sigma_f ** 2)[:, :, None, None] * torch.exp(-2 * norm)

        # Add some noise to the diagonal to make the cholesky work.
        kernel += (sigma_noise ** 2) * torch.eye(num_total_points).to(self.device)

        return kernel

    def generate_curves(self, include_context = True):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.
        Args:
            include_context:  Whether to include context data to target data, useful for NP, CNP, ANP
        Returns:
            A `CNPRegressionDescription` namedtuple.
        """
        num_context = np.random.randint(3, self._max_num_context)

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_total_points = 400
            num_target = num_total_points - num_context
            x_values = torch.arange(-2, 2, 0.01)[None, :, None].repeat([self._batch_size, 1, 1]).to(self.device)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = np.random.randint(3, self._max_num_context)
            num_total_points = num_context + num_target
            x_values = (torch.rand([self._batch_size, num_total_points, self._x_size])*4 -2).to(self.device)

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        if self.kernel == 'EQ':
            kernel = self._rbf_kernel(x_values) # [B, y_size, num_total_points, num_total_points]
        elif self.kernel == 'period':
            kernel = self._periodic_kernel(x_values)

        # Calculate Cholesky, using double precision for better stability:
        cholesky = torch.cholesky(kernel.type(torch.float64)).type(torch.float32) # [B, y_size, num_total_points, num_total_points]

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(cholesky, torch.rand([self._batch_size, self._y_size, num_total_points, 1]).to(self.device))

        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(-1).permute([0, 2, 1])

        if self._testing:
            # Select the targets
            target_x = x_values
            target_y = y_values

            # Select the observations
            idx = np.random.permutation(num_total_points)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]

        else:
            if include_context:
                # Select the targets which will consist of the context points as well as
                # some new target points
                target_x = x_values[:, :num_target + num_context, :]
                target_y = y_values[:, :num_target + num_context, :]
            else:
                target_x = x_values[:, num_context :num_target + num_context, :]
                target_y = y_values[:, num_context :num_target + num_context, :]
            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            y_target=target_y,
            num_total_points= num_target + num_context,
            num_context_points=num_context)