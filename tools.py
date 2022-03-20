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
    required for use with the rest of this package. (constrain, forward, and log_prob methods)
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
    implimented log_prob() and constrain() methods.
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

def ECDF(x: torch.Tensor, dim: int = 0):
    '''
    Finds empirical cumulative distribution function of provided data "x" along dimension "dim".
    '''
    x = torch.sort(x.flatten(dim), dim=dim).values
    n = x.shape[-1]
    cum = torch.arange(1, n + 1).to(x.device) / n
    cum = cum.repeat(*x.shape[0:-1], 1)  # one for each univariate sample
    return torch.cat((x.unsqueeze(dim), cum.unsqueeze(dim)), dim)
