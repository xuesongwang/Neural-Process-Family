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

    def _matern_kernel(self):
        # num_total_points = xdata.size(1)
        # # Expand and take the difference
        # xdata1 = torch.unsqueeze(xdata, dim=1)  # [B, 1, num_total_points, x_size]
        # xdata2 = torch.unsqueeze(xdata, dim=2)  # [B, num_total_points, 1, x_size]
        #
        # d = 4 * torch.abs(xdata1 - xdata2) # [B, num_total_points, num_total_points, x_size]
        # d = torch.sum(d, dim=-1).unsqueeze(dim=1)  # [B, y_size, num_total_points, num_total_points]
        # kernel = (1 + 4*5**0.5*d + 5.0/3.0*d**2) * torch.exp(-5**(0.5)*d)
        # # kernel += (sigma_noise ** 2) * torch.eye(num_total_points).to(self.device)
        import stheno.torch as stheno
        import stheno as sth
        kernel = stheno.Matern52().stretch(0.25)
        gp = stheno.GP(kernel, graph=sth.Graph())
        return gp

    def generate_with_Matern(self, gp):
        num_points = 200
        x_all = np.linspace(-2., 2., num_points)
        y_all = gp(x_all).sample()
        #     y_all = gp_.sample(x_all)
        x_all = torch.tensor(x_all, dtype=torch.float)[None, :, None]
        y_all = torch.tensor(y_all, dtype=torch.float).unsqueeze(0)
        self._batch_size = 1
        return x_all, y_all

    def generate_curves(self, include_context = True, x_min = -2, x_max = 2, sort=False):
        """Builds the op delivering the data.

        Generated functions are `float32` with x values between -2 and 2.
        Args:
            include_context:  Whether to include context data to target data, useful for NP, CNP, ANP
        Returns:
            A `NPRegressionDescription` namedtuple.
        """
        num_context = np.random.randint(3, self._max_num_context)

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.
        if self._testing:
            num_total_points = 400
            num_target = num_total_points - num_context
            x_values = torch.arange(x_min, x_max, 0.01)[None, :, None].repeat([self._batch_size, 1, 1]).to(self.device)
        # During training the number of target points and their x-positions are
        # selected at random
        else:
            num_target = np.random.randint(3, self._max_num_context)
            num_total_points = num_context + num_target
            x_values = (torch.rand([self._batch_size, num_total_points, self._x_size])*(x_max - x_min) + x_min).to(self.device)
        if sort == True:
            x_values, index_sorted = torch.sort(x_values, dim=1)
        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        if self.kernel == 'EQ':
            kernel = self._rbf_kernel(x_values) # [B, y_size, num_total_points, num_total_points]
        elif self.kernel == 'period':
            kernel = self._periodic_kernel(x_values)
        elif self.kernel == 'matern':
            kernel = self._matern_kernel()

        if self.kernel != 'matern':
            # Calculate Cholesky, using double precision for better stability:
            # cholesky = torch.linalg.cholesky(kernel.type(torch.float64)).type(torch.float32) # [B, y_size, num_total_points, num_total_points]
            cholesky = np.linalg.cholesky(kernel.cpu().numpy())
            cholesky = torch.tensor(cholesky).type(torch.FloatTensor).to(kernel.device)
            # Sample a curve
            # [batch_size, y_size, num_total_points, 1]
            y_values = torch.matmul(cholesky, torch.rand([self._batch_size, self._y_size, num_total_points, 1]).to(self.device))
            # [batch_size, num_total_points, y_size]
            y_values = y_values.squeeze(-1).permute([0, 2, 1])
        else:
            x_values, y_values = self.generate_with_Matern(kernel)
            idx = np.random.permutation(x_values.size(1))
            x_values = x_values[:, idx, :]
            y_values = y_values[:, idx, :]

        # scale
        # scale = 2*np.random.rand() + 1 #scale by a factor from [1, 3)
        # bias = 3*np.random.rand() #bias by [0,3)
        scale = 1
        bias = 0
        y_values = y_values * scale + bias

        if self._testing:
            # Select the observations
            idx = np.random.permutation(num_total_points)
            context_x = x_values[:, idx[:num_context], :]
            context_y = y_values[:, idx[:num_context], :]
            target_x = x_values[:, idx[num_context:], :]
            target_y = y_values[:, idx[num_context:], :]
        else:

            if include_context:
                # Select the targets which constitute the context points as well as
                # some new target points
                target_x = x_values[:, :(num_target + num_context), :]
                target_y = y_values[:, :(num_target + num_context), :]
            else:
                target_x = x_values[:, num_context :(num_target + num_context), :]
                target_y = y_values[:, num_context :(num_target + num_context), :]
            # Select the observations
            context_x = x_values[:, :num_context, :]
            context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            y_target=target_y,
            num_total_points= num_target + num_context,
            num_context_points=num_context)

    def generate_temporal_curves(self, max_num_context = 10, include_context = True):
        """Builds the op delivering the data.

                Generated functions are `float32` with x values between -2 and 0 for context data, and 0 to 2 for target data.
                Args:
                    max_num_context:  we set max_num_context to be smaller(25) than generate curve for sequential data
                    include_context:  Whether to include context data to target data, useful for NP, CNP, ANP
                Returns:
                    A `NPRegressionDescription` namedtuple.
                """
        self._max_num_context = max_num_context
        num_context =  self._max_num_context # fixed sequence

        # the number of target points and their x-positions are
        # selected at random
        num_target = self._max_num_context #fixed sequence

        num_total_points = num_context + num_target
        x_context = -2 * torch.rand([self._batch_size, num_context, self._x_size]).to(self.device)
        x_target = 2 * torch.rand([self._batch_size, num_target, self._x_size]).to(self.device)
        x_values = torch.cat([x_context, x_target], dim=1)

        # Pass the x_values through the Gaussian kernel
        # [batch_size, y_size, num_total_points, num_total_points]
        if self.kernel == 'EQ':
            kernel = self._rbf_kernel(x_values)  # [B, y_size, num_total_points, num_total_points]
        elif self.kernel == 'period':
            kernel = self._periodic_kernel(x_values)

        # Calculate Cholesky, using double precision for better stability:
        # cholesky = torch.linalg.cholesky(kernel.type(torch.float64)).type(
        #     torch.float32)  # [B, y_size, num_total_points, num_total_points]
        cholesky = np.linalg.cholesky(kernel.cpu().numpy())
        cholesky = torch.tensor(cholesky).type(torch.FloatTensor).to(kernel.device)

        # Sample a curve
        # [batch_size, y_size, num_total_points, 1]
        y_values = torch.matmul(cholesky,
                                torch.rand([self._batch_size, self._y_size, num_total_points, 1]).to(self.device))

        # [batch_size, num_total_points, y_size]
        y_values = y_values.squeeze(-1).permute([0, 2, 1])

        if include_context:
            # Select the targets which constitute the context points as well as
            # some new target points
            target_x = x_values[:, :num_target + num_context, :]
            target_y = y_values[:, :num_target + num_context, :]
        else:
            target_x = x_values[:, num_context:num_target + num_context, :]
            target_y = y_values[:, num_context:num_target + num_context, :]
        # Select the observations
        context_x = x_values[:, :num_context, :]
        context_y = y_values[:, :num_context, :]

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            y_target=target_y,
            num_total_points=num_target + num_context,
            num_context_points=num_context)
