import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

device = "cuda"


class Unbounded_Metalog_Model(nn.Module):
    """
    An implimentation of unbounded metalog models.
    """

    def __init__(self, init_a: torch.Tensor = None):
        super(Unbounded_Metalog_Model, self).__init__()

        self.a = nn.Parameter(init_a, requires_grad=True)
        self.n = self.a.shape[-1]

        ### Define basis functions for QF (quantile function):
        def qg1(y, i):
            """first basis function"""
            return torch.ones_like(y)

        def qg2(y, i):
            """second basis function"""
            return torch.log(y / (1 - y))

        def qg3(y, i):
            """third basis function"""
            return (y - 0.5) * torch.log(y / (1 - y))

        def qg4(y, i):
            """fourth basis function"""
            return y - 0.5

        def qgj_odd(y, j):
            """nth odd basis function (after third)"""
            j += 1
            assert (j % 2 != 0) and (j >= 5)
            return (y - 0.5).pow((j - 1) / 2)

        def qgj_even(y, j):
            """nth even basis function (after fourth)"""
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

        ### Define basis functions for derivative of quantile function in terms of cumulative
        ### probability. (^ derivative of quantile function):
        def dqg1(y, i):
            """first basis function"""
            return torch.zeros_like(y)

        def dqg2(y, i):
            """second basis function"""
            return 1 / (y * (1 - y))

        def dqg3(y, i):
            """third basis function"""
            return (y - 1 / 2) / (y * (1 - y)) + torch.log(y / (1 - y))

        def dqg4(y, i):
            """fourth basis function"""
            return torch.ones_like(y)

        def dqgj_odd(y, j):
            """nth odd basis function (after third)"""
            j += 1
            assert (j % 2 != 0) and (j >= 5)
            return ((j - 1) / 2) * (y - 1 / 2).pow((j - 3) / 2)

        def dqgj_even(y, j):
            """nth even basis function (after fourth)"""
            j += 1
            assert (j % 2 == 0) and (j >= 6)
            return (y - 1 / 2).pow(j / 2 - 1) / (y * (1 - y)) + (j / 2 - 1) * (
                y - 1 / 2
            ).pow(j / 2 - 2) * torch.log(y / (1 - y))

        # Start derivative quantile basis functions:
        self.dqf_basis_functions = [dqg1, dqg2, dqg3, dqg4]
        # Additional dqf basis functions as needed:
        self.dqf_basis_functions = self.dqf_basis_functions + [
            dqgj_odd if x % 2 == 0 else dqgj_even for x in range(self.n - 4)
        ]
        # Trim as needed:
        self.dqf_basis_functions = self.dqf_basis_functions[: self.n]

    def constrain(self):
        """Coefficients are unconstrained in this case."""
        pass

    def quantile(self, y):
        """
        Quantile of cumulative probability "y".  (returns x-position of cumulative probability "y".
        This is an inverse CDF)
        """
        x_values = sum(
            [
                self.a[:, idx].unsqueeze(-1) * f(y, idx)
                for idx, f in enumerate(self.qf_basis_functions)
            ]
        )
        return x_values

    def derivative_quantile(self, y):
        """
        Derivative of quantile as function of cumulative probability "y".
        (AKA: quantile density function.)
        """
        return sum(
            [
                self.a[:, idx].unsqueeze(-1) * f(y, idx)
                for idx, f in enumerate(self.dqf_basis_functions)
            ]
        )

    def prob_ito_cumprob(self, y):
        """Probability density in terms of cumulative probability "y"."""
        return self.derivative_quantile(y).pow(-1)

    def prob(self, x, iters=64):
        """
        Approximates probability density at a batch of tensors "x" by asymptotically bounded
        approach. There is currently no known closed-form inverse metalog.
        """
        eps = 1e-7
        cum_y_guess = torch.ones_like(x) * 1 / 3

        lr = 1 / 3
        old_x_guess = self.quantile(cum_y_guess)  # initial
        old_diff = 0  # initial
        adj = torch.tensor([1]).to(self.a.device) / x.shape[1]  # initial
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
        """Approximates log of probability density at a batch of tensors "x"."""
        return torch.log(self.prob(x))

    def estimate_entropy(self, steps=256):
        """Estimates shannon entropy of the distribution in nats by numeric integration."""
        self.a.data = self.a.data.double()  # increase precision
        eps = 1e-7
        a = eps  # lower integration bound
        b = 1 - eps  # upper integration bound
        cum_y_tics = torch.Tensor(np.linspace(a, b, steps)).double().to(self.a.device)
        # shape for batch and channel support;
        cum_y_tics = cum_y_tics.repeat(self.a.shape[0], 1)

        qp_tics = self.derivative_quantile(cum_y_tics)
        entropy = torch.trapz(torch.nan_to_num(torch.log(qp_tics), 0), cum_y_tics)

        self.a.data = self.a.data.float()  # reset precision

        return entropy

    def sample(self, shape):
        """Simulates data of shape "shape" by inverse tranform sampling."""
        eps = 1e-7
        return self.quantile(
            torch.rand(shape).clamp(min=eps, max=1 - eps).to(self.a.device)
        )

    def forward(self, x):
        """
        By default: Approximates probability density at a batch of tensors "x" by asymptotically
        bounded approach. There is currently no known closed-form inverse metalog.
        """
        return self.prob(x)


def ECDF(x: torch.Tensor, dim: int = 0, reach_limits=True):
    """
    set "reach_limit" to false to calculate ECDF in a way that will not include perfect 0 or 1.
    """
    x = torch.sort(x.flatten(dim), dim=dim).values
    n = x.shape[-1]
    cum = torch.arange(1, n + 1).to(x.device) / (n + 1 - reach_limits)
    cum = cum.repeat(*x.shape[0:-1], 1)  # one for each univariate sample
    return x, cum


def Metalog_Fit_Closed_Form(model, data, weights=None):
    """Fits the parameters of the model to the data by a closed-form linear least-squares method."""
    ecdf = ECDF(data, dim=1, reach_limits=False)
    x, y = ecdf
    x = x.float()
    y = y.float()

    Y_cols = [f(y, idx) for idx, f in enumerate(model.qf_basis_functions)]
    Y = torch.stack(Y_cols, -1)
    #  print("Y:", Y)
    a = torch.bmm(
        torch.linalg.solve(torch.bmm(Y.transpose(1, 2), Y), Y.transpose(1, 2)),
        x.unsqueeze(-1),
    ).flatten(1)
    model.a.data = a


samples_shape = [4096]
A = Normal(
    torch.zeros(samples_shape).to(device), torch.ones(samples_shape).to(device)
)  # known skew = 0, known kurt = 0
a = A.sample()

eB = Normal(
    torch.zeros(samples_shape).to(device) - 10,
    torch.ones(samples_shape).to(device) * 1,
)
b = 0 * a + eB.sample()

eC = Normal(
    torch.zeros(samples_shape).to(device) + 2,
    torch.ones(samples_shape).to(device) * 1 / 2,
)
c = -a.pow(2) + eC.sample()

eD = Normal(
    torch.zeros(samples_shape).to(device) - 3,
    torch.ones(samples_shape).to(device) * 1/4,
)
d = a.pow(2) + eD.sample()

eE = Normal(
    torch.zeros(samples_shape).to(device) + 0,
    torch.ones(samples_shape).to(device) * 1/4,
)
e = torch.sin(a) + eE.sample()

data = torch.stack((a, b, c, d, e), 0)

print("data.shape:", data.shape)


def Metalog_Fit_Closed_Form(model, data, weights=None):
    """Fits the parameters of the model to the data by a closed-form linear least-squares method."""
    ecdf = ECDF(data, dim=1, reach_limits=False)
    x, y = ecdf
    x = x.float()
    y = y.float()

    Y_cols = [f(y, idx) for idx, f in enumerate(model.qf_basis_functions)]
    Y = torch.stack(Y_cols, -1)
    #  print("Y:", Y)
    if weights is None:
        a = torch.bmm(
            torch.linalg.solve(torch.bmm(Y.transpose(1, 2), Y), Y.transpose(1, 2)),
            x.unsqueeze(-1),
        ).flatten(1)
    else:
        a = torch.bmm(
            torch.linalg.solve(
                torch.bmm(Y.transpose(1, 2), torch.bmm(weights, Y)), Y.transpose(1, 2)
            ),
            torch.bmm(weights, x.unsqueeze(-1)),
        ).flatten(1)
    model.a.data = a


import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")

def polynomial_fit(x, y, num_terms=5, weights=None):
    X_cols = [x.pow(n) for n in range(num_terms)]
    X = torch.stack(X_cols, -1)

    if weights is None:
        a = torch.bmm(
            torch.linalg.solve(torch.bmm(X.transpose(1, 2), X), X.transpose(1, 2)),
            y.unsqueeze(-1),
        ).flatten(1)
    else:
        a = torch.bmm(
            torch.linalg.solve(
                torch.bmm(X.transpose(1, 2), torch.bmm(weights, X)), X.transpose(1, 2)
            ),
            torch.bmm(weights, y.unsqueeze(-1)),
        ).flatten(1)
    return a

def mutual_information(data, idx, num_terms=5, weights=None):
    channels = data.shape[0]

    initial_model = Unbounded_Metalog_Model(init_a=torch.zeros([channels, 7])).to(
        device
    )
    Metalog_Fit_Closed_Form(initial_model, data)
    initial_entropy = initial_model.estimate_entropy()

    ##########################
    ### model relationship ###
    x = data[idx].repeat(channels, 1)
    y = data

    X_cols = [x.pow(n) for n in range(num_terms)]
    X = torch.stack(X_cols, -1)

    if weights is None:
        a = torch.bmm(
            torch.linalg.solve(torch.bmm(X.transpose(1, 2), X), X.transpose(1, 2)),
            y.unsqueeze(-1),
        ).flatten(1)
    else:
        a = torch.bmm(
            torch.linalg.solve(
                torch.bmm(X.transpose(1, 2), torch.bmm(weights, X)), X.transpose(1, 2)
            ),
            torch.bmm(weights, y.unsqueeze(-1)),
        ).flatten(1)

    #####################
    ### for business: ###

    y = torch.mm(a, X[0, :].T)

    x = data[idx]
    #####################

    #################
    ### for show: ###

    #  y = torch.mm(a, X)
    print("y.shape", y.shape)

    x, indices = data[idx].sort()
    print("x.shape", x.shape)
    print("indices.shape", indices.shape)
    rev_indices = indices.sort().indices
    rev_indices = rev_indices.repeat(y.shape[0], 1)
    print("rev_indices.shape", rev_indices.shape)
    y = y.scatter(1, rev_indices, y)
    print("y.shape", y.shape)

    for i in range(channels):
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(data[idx].cpu().detach().numpy(), data[i].cpu().detach().numpy(), ".")
        ax.plot(x.cpu().detach().numpy(), y[i].cpu().detach().numpy())
        plt.show()
    #################

    y = torch.mm(a, X[0, :].T)

    x = data[idx]
    ##########################


    ### account for relationship:
    data = data - y
    y = y - y

    # display data after relationship removal:
    for i in range(channels):
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(x.cpu().detach().numpy(), data[i].cpu().detach().numpy(), ".")
        ax.plot(x.cpu().detach().numpy(), y[i].cpu().detach().numpy(), ".")
        plt.show()

    # calculate entropy after relationship removal:
    final_model = Unbounded_Metalog_Model(init_a=torch.zeros([channels, 7])).to(
        device
    )
    Metalog_Fit_Closed_Form(final_model, data)
    final_entropy = final_model.estimate_entropy()

    print("initial_entropy:", initial_entropy.tolist())
    print("final_entropy:", final_entropy.tolist())
    print("entropy difference:", (final_entropy - initial_entropy).tolist())

mutual_information(data, 0)
