import torch
import torch.nn as nn
from torch.distributions import Normal

def skewness_fn(x, dim=1):
    '''Calculates skewness of data "x" along dimension "dim".'''
    std, mean = torch.std_mean(x, dim)
    n = torch.Tensor([x.shape[dim]]).to(x.device)
    eps = 1e-6  # for stability

    sample_bias_adjustment = torch.sqrt(n * (n - 1)) / (n - 2)
    skewness = sample_bias_adjustment * (
        (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(3), dim) / n)
        / std.pow(3).clamp(min=eps)
    )
    return skewness


def kurtosis_fn(x, dim=1):
    '''Calculates kurtosis of data "x" along dimension "dim".'''
    std, mean = torch.std_mean(x, dim)
    n = torch.Tensor([x.shape[dim]]).to(x.device)
    eps = 1e-6  # for stability

    sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
    kurtosis = sample_bias_adjustment * (
        (n + 1)
        * (
            (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(4), dim) / n)
            / std.pow(4).clamp(min=eps)
        )
        - 3 * (n - 1)
    )
    return kurtosis

def bimodality_index(x, dim=1):
    '''
    Used to detect bimodality (or multimodality) of dataset(s) given a tensor "x" containing the
    data and a dimension "dim" along which to calculate.  The logic behind this index is that a
    bimodal (or multimodal) distribution with light tails will have very low kurtosis, an asymmetric
    character, or both – all of which increase this index.  The smaller this value is the more
    likely the data are to follow a unimodal distribution.  As a rule: if return value ≤ 0.555
    (bimodal index for uniform distribution), the data are considered to follow a unimodal
    distribution. Otherwise, they follow a bimodal or multimodal distribution.
    '''
    # calculate standard deviation and mean of dataset(s)
    std, mean = torch.std_mean(x, dim)
    # get number of samples in dataset(s)
    n = torch.Tensor([x.shape[dim]]).to(x.device)
    eps = 1e-6  # for stability

    # calculate skewness:
    # repeating most of the skewness function here to avoid recomputation of standard devation and mean
    sample_bias_adjustment = torch.sqrt(n * (n - 1)) / (n - 2)
    skew = sample_bias_adjustment * (
        (torch.sum((x.T - mean.unsqueeze(dim).T).T.pow(3), dim) / n)
        / std.pow(3).clamp(min=eps)
    )

    # calculate kurtosis:
    # repeating most the kurtosis function here to avoid recomputation of standard devation and mean
    sample_bias_adjustment = (n - 1) / ((n - 2) * (n - 3))
    kurt = sample_bias_adjustment * (
        (n + 1)
        * (
            (torch.sum((z.T - mean.unsqueeze(dim).T).T.pow(4), dim) / n)
            / std.pow(4).clamp(min=eps)
        )
        - 3 * (n - 1)
    )

    # calculate bimodality index:
    BC = (skew.pow(2) + 1) / (kurt + 3 * ((n - 2).pow(2) / ((n - 2) * (n - 3))))

    return BC

def KernelDensityEstimate(
    data,
    x_tics=None,
    start=-9,
    end=9,
    kernel=Normal(loc=0, scale=1),
    bandwidth_adjustment=1,
    dim=1,
):
    '''Estimates the probability density function of a batch of data.'''
    # convert to positive index (important for unsqueezing)
    if dim < 0:
        dim = len(data.shape) + dim
        if dim > (len(data.shape) - 1) or dim < 0:
            raise IndexError

    def kde_prob(
        data,
        x,
        dim=dim,
        kernel=Normal(loc=0, scale=1),
        bandwidth_adjustment=1,
    ):
        '''
        Returns the probability of the items in tensor 'x' according to the PDF estimated by a KDE.
        This function is memory intensive.
        '''
        data = data.flatten(dim)
        n = data.shape[dim]
        silvermans_factor = ((4 * torch.std(data, dim).pow(5)) / (3 * n)).pow(1 / 5)

        bw = silvermans_factor * bandwidth_adjustment

        bw = bw.view(
            *bw.shape, *[1 for _ in range(len(data.shape) + 1 - len(bw.shape))]
        )
        a = data.unsqueeze(dim) - x.unsqueeze(1)
        a = a / bw
        a = kernel.log_prob(a)
        a = torch.exp(a)
        a = bw ** (-1) * a
        a = a.sum(dim=dim + 1)
        prob = a / n

        return prob

    if x_tics is None:
        assert start and end
        assert end > start
        a = max(torch.min(data).item(), start)  # lower integration bound
        b = min(torch.max(data).item(), end)  # upper integration bound
        x_tics = torch.Tensor(np.linspace(a, b, steps)).to(data.device)
    else:
        assert isinstance(x_tics, torch.Tensor)
        x_tics = x_tics.to(data.device)
    x_tics.requires_grad = True
    kde_y_tics = kde_prob(
        data,
        x_tics,
        kernel=kernel,
        bandwidth_adjustment=bandwidth_adjustment,
    )
    return kde_y_tics

class Normal_Model(nn.Module):
    '''
    Example of a module for modeling a probability distribution. This is set up with all pieces
    required for use with the rest of this package. (initial parameters; as well as implimented
    constrain, forward, and log_prob methods)
    '''
    def __init__(
        self,
        init_mean: torch.Tensor = torch.Tensor([0]),
        init_std: torch.Tensor = torch.Tensor([1]),
    ):
        super(Normal_Model, self).__init__()
        self.mean = nn.Parameter(init_mean, requires_grad=True)
        self.std = nn.Parameter(init_std, requires_grad=True)
        # constant
        self.ln2p = nn.Parameter(
            torch.log(2 * torch.Tensor([torch.pi])), requires_grad=False
        )

    def constrain(self):
        '''
        Method to run on "constrain" step of training. Easiest method for optimization under
        constraint is Projection Optimization by simply clamping parameters to bounds after each
        update. This is certainly not the most efficent way, but it gets the job done.
        '''
        #  can't have negative standard deviation so lets prevent that:
        eps = 1e-6
        self.std.data = model.std.data.clamp(min=eps)

    def log_prob(self, x):
        '''
        Returns the log probability of the items in tensor 'x' according to the probability
        distribution of the module.
        '''
        return (
            -torch.log(self.std.unsqueeze(-1))
            - (self.ln2p / 2)
            - ((x - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)).pow(2) / 2
        )

    def forward(self, x):
        '''Returns the probability of the items in tensor 'x' according to the probability distribution of the module.'''
        return self.log_prob(x).exp()

def MLE_Fit(model, data, dim=1, lr=5e-2, iters=250):
    '''
    Fits the parameters of the provided model to the provided data. Provided model must have
    implimented log_prob() and constrain() methods, and paraters set to some initial value.
    '''
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    #  print("model parameters:", [x for x in model.parameters()])
    #  data = data.flatten(dim)
    for i in range(iters):
        nll = -torch.sum(model.log_prob(data))
        nll.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.constrain()

def ECDF(x: torch.Tensor, dim: int = 0, reach_limits=True):
    """
    set "reach_limit" to false to calculate ECDF in a way that will not include perfect 0 or 1.
    """
    x = torch.sort(x.flatten(dim), dim=dim).values
    n = x.shape[-1]
    cum = torch.arange(1, n + 1).to(x.device) / (n + 1 - reach_limits)
    cum = cum.repeat(*x.shape[0:-1], 1)  # one for each univariate sample
    return x, cum

class Unbounded_Metalog_Model(nn.Module):
    def __init__(
        self,
        init_a: torch.Tensor = None,
    ):
        super(Unbounded_Metalog_Model, self).__init__()

        self.a = nn.Parameter(init_a, requires_grad=True)
        self.n = self.a.shape[-1]

        ### Define basis functions for QF (quantile function):
        def qg1(y, i):
            '''first basis function'''
            return torch.ones_like(y)

        def qg2(y, i):
            '''second basis function'''
            return torch.log(y / (1 - y))

        def qg3(y, i):
            '''third basis function'''
            return (y - 0.5) * torch.log(y / (1 - y))

        def qg4(y, i):
            '''fourth basis function'''
            return y - 0.5

        def qgj_odd(y, j):
            '''nth odd basis function (after third)'''
            j += 1
            assert (j % 2 != 0) and (j >= 5)
            return (y - 0.5).pow((j - 1) / 2)

        def qgj_even(y, j):
            '''nth even basis function (after fourth)'''
            j += 1
            assert (j % 2 == 0) and (j >= 6)
            return torch.log(y / (1 - y)) * (y - 0.5).pow(j / 2 - 1)

        # Start QF basis functions:
        self.qf_basis_functions = [qg1, qg2, qg3, qg4]
        # Additional inverse cdf basis functions as needed:
        self.qf_basis_functions = self.qf_basis_functions + [
            qgj_odd if x % 2 == 0 else qgj_even for x in range(self.n - 4)
        ]
        # Trim as needed:
        self.qf_basis_functions = self.qf_basis_functions[: self.n]

        ### Define basis functions for inverse PDF in terms of cumulative probability
        ### (^ derivative of quantile function):
        def dqg1(y, i):
            '''first basis function'''
            return torch.zeros_like(y)

        def dqg2(y, i):
            '''second basis function'''
            return 1 / (y * (1 - y))

        def dqg3(y, i):
            '''third basis function'''
            return (y - 1 / 2) / (y * (1 - y)) + torch.log(y / (1 - y))

        def dqg4(y, i):
            '''fourth basis function'''
            return torch.ones_like(y)

        def dqgj_odd(y, j):
            '''nth odd basis function (after third)'''
            j += 1
            assert (j % 2 != 0) and (j >= 5)
            return ((j - 1) / 2) * (y - 1 / 2).pow((j - 3) / 2)

        def dqgj_even(y, j):
            '''nth even basis function (after fourth)'''
            j += 1
            assert (j % 2 == 0) and (j >= 6)
            return (y - 1 / 2).pow(j / 2 - 1) / (y * (1 - y)) + (j / 2 - 1) * (
                y - 1 / 2
            ).pow(j / 2 - 2) * torch.log(y / (1 - y))

        # Start PDF basis functions:
        self.ipdf_basis_functions = [dqg1, dqg2, dqg3, dqg4]
        # Additional ipdf basis functions as needed:
        self.ipdf_basis_functions = self.ipdf_basis_functions + [
            dqgj_odd if x % 2 == 0 else dqgj_even for x in range(self.n - 4)
        ]
        # Trim as needed:
        self.ipdf_basis_functions = self.ipdf_basis_functions[: self.n]

    def constrain(self):
        '''Coefficients are unconstrained in this case.'''
        pass

    def quantile(self, y):
        '''
        Quantile of cumulative probability "y".  (returns x-position of cumulative probability "y".
        This is an inverse CDF)
        '''
        x_tics = sum(
            [
                self.a[:, idx].unsqueeze(-1) * f(y, idx)
                for idx, f in enumerate(self.qf_basis_functions)
            ]
        )
        return x_tics

    def prob_ito_cumprob(self, y):
        '''Probability density in terms of cumulative probability "y".'''
        return sum(
            [
                self.a[:, idx].unsqueeze(-1) * f(y, idx)
                for idx, f in enumerate(self.ipdf_basis_functions)
            ]
        ).pow(
            -1
        )  # for reciprocal of sum of terms

    def prob(self, x, iters=64):
        '''
        Approximates probability density at a batch of tensors "x" by asymptotically bounded
        approach. There is currently no known closed-form inverse metalog.
        '''
        eps = 1e-7
        cum_y_guess = torch.ones_like(x) * 1 / 3

        lr = 1/3
        old_x_guess = self.quantile(cum_y_guess)  # initial
        old_diff = 0  # initial
        adj = torch.tensor([1]).to(x.device) / x.shape[1]  # initial
        for i in range(iters):
            cum_y_guess += adj
            x_guess = self.quantile(cum_y_guess)
            diff = x - x_guess
            #  print(f"mean squared diff {i}:", (torch.sum(diff.pow(2))/diff.shape[1]).item())
            max_adj = (
                torch.heaviside(diff, torch.Tensor([0]).to(cum_y_guess)).clamp(
                    min=eps, max=1 - eps
                )
                - cum_y_guess
            )
            adj = max_adj * torch.tanh(diff.pow(2)) * lr

        density = self.prob_ito_cumprob(cum_y_guess)
        density = torch.nan_to_num(density, nan=0)
        return density

    def log_prob(self, x):
        '''Approximates log of probability density at a batch of tensors "x".'''
        return torch.log(self.prob(x))

    def estimate_entropy(self, steps=256):
        '''Estimates shannon entropy of the distribution in nats by numeric integration.'''
        eps = 1e-7
        a = eps  # lower integration bound
        b = 1 - eps  # upper integration bound
        cum_y_tics = torch.Tensor(np.linspace(a, b, steps)).to(device)
        x_tics = self.quantile(cum_y_tics)
        p_tics = self.prob_ito_cumprob(cum_y_tics)
        entropy = -torch.trapz(p_tics*torch.log(p_tics), x_tics)
        return entropy

    def sample(self, shape: torch.Tensor.shape):
        '''Simulates data by inverse tranform.'''
        eps = 1e-7
        return self.quantile(torch.rand(shape).clamp(min=eps, max=1-eps))

    def forward(self, x):
        '''
        By default: Approximates probability density at a batch of tensors "x" by asymptotically
        bounded approach. There is currently no known closed-form inverse metalog.
        '''
        return self.prob(x)

def Metalog_Fit_Closed_Form(model, data):
    """
    Fits the parameters of the metalog model, "model", to sources of data in "data", by a closed-form
    linear least-squares method.
    This function supports batching for fitting many datasets at once and expects data in batched
    form (with at least 2 dimensional shape). First dimension of data must match first dimension of
    model coefficients "a". If first dimension > 1 (batchsize > 1), this function will fit a number
    of sets of coefficients, namely: one set of coefficients in the provided metalog model for each
    dataset, where the first-dimension or "batch-size" of "data" indicates the number of independent
    datasets.
    """
    ecdf = ECDF(data, dim=1, reach_limits=False)
    x, y = ecdf
    x = x.float()
    y = y.float()

    Y_cols = [f(y, idx) for idx, f in enumerate(model.qf_basis_functions)]
    Y = torch.stack(Y_cols, -1)
    a = torch.bmm(
        torch.linalg.solve(torch.bmm(Y.transpose(1, 2), Y), Y.transpose(1, 2)),
        x.unsqueeze(-1),
    ).flatten(1)
    model.a.data = a

