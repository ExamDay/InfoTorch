import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

import matplotlib.pyplot as plt
import seaborn as sns
import patchworklib as pw

sns.set()
plt.style.use("seaborn-whitegrid")
pw.overwrite_axisgrid()

device = "mps"


class Metalog_Model(nn.Module):
    """
    An implimentation of bounded metalog models.
    Takes an matrix of coefficients "a" of shape [M, N], where M is the number of distributions to
    model, and N is the number of terms in each approximant, as well as a matrix of bounds "bounds"
    of shape [M, 2] where M is again the number of distributions to model and the second dimension
    holds bounds for each distribution like [lower bound, upper bound]
    """

    def __init__(
        self,
        init_a: torch.Tensor = None,
        lower_bounds: torch.Tensor = None,
        upper_bounds: torch.Tensor = None,
    ):
        super(Metalog_Model, self).__init__()

        self.a = nn.Parameter(init_a, requires_grad=True)
        self.n = self.a.shape[-1]
        if lower_bounds is None:
            self.l_bounds = None
        else:
            self.l_bounds = nn.Parameter(lower_bounds, requires_grad=False)
        if upper_bounds is None:
            self.u_bounds = None
        else:
            self.u_bounds = nn.Parameter(upper_bounds, requires_grad=False)

        self.apply_bound = self.bounding()
        self.apply_d_bound = self.bounding()

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
        unbound_qf_basis_functions = [qg1, qg2, qg3, qg4]
        # Additional quantile basis functions as needed:
        unbound_qf_basis_functions = unbound_qf_basis_functions + [
            qgj_odd if x % 2 == 0 else qgj_even for x in range(self.n - 4)
        ]
        # Trim as needed:
        unbound_qf_basis_functions = unbound_qf_basis_functions[: self.n]

        # Apply bounding functions
        self.qf_basis_functions = [
            self.bounding(x, self.l_bounds, self.u_bounds)
            for x in unbound_qf_basis_functions
        ]
        #  self.qf_basis_functions = [
        #      lambda x, idx: bf(x, idx) for idx, bf in enumerate(bounding_functions)
        #  ]

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

        # Start reciprocal derivative quantile basis functions:
        unbound_dqf_basis_functions = [dqg1, dqg2, dqg3, dqg4]
        # Additional dqf basis functions as needed:
        unbound_dqf_basis_functions = unbound_dqf_basis_functions + [
            dqgj_odd if x % 2 == 0 else dqgj_even for x in range(self.n - 4)
        ]
        # Trim as needed:
        unbound_dqf_basis_functions = unbound_dqf_basis_functions[: self.n]

        # Apply derivative bounding functions
        self.dqf_basis_functions = [
            self.d_bounding(x, y, self.l_bounds, self.u_bounds)
            for x, y in zip(unbound_qf_basis_functions, unbound_dqf_basis_functions)
        ]
        #  self.dqf_basis_functions = [
        #      lambda x, idx: dbf(x, idx) for idx, dbf in enumerate(d_bounding_functions)
        #  ]

    class bounding:
        def __init__(self, basis=None, lower_bounds=None, upper_bounds=None):
            self.basis = basis
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds

            def full_bound(y):
                expM = torch.exp(y)
                return (lower_bounds + upper_bounds * expM) / (1 + expM)

            def semi_bound_l(y):
                return self.lower_bounds + torch.exp(y)

            def semi_bound_u(y):
                return self.upper_bounds - torch.exp(y)

            if (self.upper_bounds is not None) and (self.lower_bounds is not None):
                self.adjust_input = lambda x: x
                self.wrap = full_bound
            elif self.lower_bounds is not None:
                self.adjust_input = lambda x: x
                self.wrap = semi_bound_l
            elif self.upper_bounds is not None:
                self.adjust_input = lambda x: 1 - x
                self.wrap = semi_bound_u
            else:
                self.adjust_input = lambda x: x
                self.wrap = lambda x: x

        def __call__(self, x, idx):
            return self.wrap(self.basis(self.adjust_input(x), idx))

    class d_bounding:
        def __init__(self, basis, d_basis, lower_bounds=None, upper_bounds=None):
            self.basis = basis
            self.d_basis = d_basis
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds

            def full_bound(y, idx):
                expM = torch.exp(self.basis(y, idx))
                return self.d_basis(y, idx) * (
                    (1 + expM).pow(2) / ((upper_bounds - lower_bounds) * expM)
                )

            def semi_bound(y, idx):
                print("self.d_basis:", self.d_basis)
                A = self.d_basis(y, idx)
                B = -self.basis(y, idx)
                B = torch.exp(B)
                return A * B
                #  return self.d_basis(y, idx) * torch.exp(-self.basis(y, idx))

            if (self.upper_bounds is not None) and (self.lower_bounds is not None):
                self.wrap = full_bound
            elif (self.lower_bounds is not None) or (self.upper_bounds is not None):
                self.wrap = semi_bound
            else:
                self.wrap = lambda x, y: self.d_basis(x, y)

        def __call__(self, x, idx):
            return self.wrap(x, idx)

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
        #  self.a.data = self.a.data.double()  # increase precision
        eps = 1e-7
        a = eps  # lower integration bound
        b = 1 - eps  # upper integration bound
        #  cum_y_tics = torch.Tensor(np.linspace(a, b, steps)).double().to(self.a.device)
        cum_y_tics = torch.Tensor(np.linspace(a, b, steps)).to(self.a.device)
        # shape for batch and channel support;
        cum_y_tics = cum_y_tics.repeat(self.a.shape[0], 1)

        qp_tics = self.derivative_quantile(cum_y_tics)
        entropy = torch.trapz(torch.nan_to_num(torch.log(qp_tics), 0), cum_y_tics)

        #  self.a.data = self.a.data.float()  # reset precision

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
        #  self.a.data = self.a.data.double()  # increase precision
        eps = 1e-7
        a = eps  # lower integration bound
        b = 1 - eps  # upper integration bound
        #  cum_y_tics = torch.Tensor(np.linspace(a, b, steps)).double().to(self.a.device)
        cum_y_tics = torch.Tensor(np.linspace(a, b, steps)).to(self.a.device)
        # shape for batch and channel support;
        cum_y_tics = cum_y_tics.repeat(self.a.shape[0], 1)

        qp_tics = self.derivative_quantile(cum_y_tics)
        entropy = torch.trapz(torch.nan_to_num(torch.log(qp_tics), 0), cum_y_tics)

        #  self.a.data = self.a.data.float()  # reset precision

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
    if device == "mps":
        # can't do sorting on mps yet so:
        x = x.to("cpu")  # <- move to cpu
        x = torch.sort(x.flatten(dim), dim=dim).values
        n = x.shape[-1]
        cum = torch.arange(1, n + 1) / (n + 1 - reach_limits)
        cum = cum.repeat(*x.shape[0:-1], 1)  # one for each univariate sample
        # move back to device:
        x = x.to(device)
        cum = cum.to(device)
    else:
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

    #  initial_model = Unbounded_Metalog_Model(init_a=torch.ones([channels, 7])).to(device)
    initial_model = Metalog_Model(
        init_a=torch.ones([channels, 7]).to(device),
        lower_bounds=torch.ones([channels, 1]).to(device) * -1000,
        upper_bounds=torch.ones([channels, 1]).to(device) * 1000,
    ).to(device)
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

    relationship = torch.mm(a, X[0, :].T)

    res_x = data[idx]

    ### account for relationship:
    residue = data - relationship
    relationship_residue = relationship - relationship

    ### PLOT:
    bricks = []
    for i in range(channels):
        #  fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

        G0 = sns.JointGrid(
            dropna=True, xlim=(-10, 10), ylim=(-10, 10), marginal_ticks=False
        )

        # plot datapoints before relationship removal:
        sns.scatterplot(
            ax=G0.ax_joint,
            x=data[idx].cpu().detach().numpy(),
            y=data[i].cpu().detach().numpy(),
        )
        # plot kde:
        sns.kdeplot(
            x=data[idx].cpu().detach().numpy(),
            linewidth=1.5,
            ax=G0.ax_marg_x,
            bw_adjust=1.25,
            fill=True,
            common_norm=True,
        )
        sns.kdeplot(
            y=data[i].cpu().detach().numpy(),
            linewidth=1.5,
            ax=G0.ax_marg_y,
            bw_adjust=1.25,
            fill=True,
            common_norm=True,
        )
        # plot relationships:
        relationship_curve_x = torch.DoubleTensor(
            np.linspace(data[idx].min().cpu(), data[idx].max().cpu(), 2048)
        )
        relationship_curve_y = sum(
            [
                coeff * relationship_curve_x.pow(idx)
                for idx, coeff in enumerate(a[i].cpu())
            ]
        )

        sns.lineplot(
            ax=G0.ax_joint,
            #  x=x.cpu().detach().numpy(),
            #  y=y[i].cpu().detach().numpy(),
            x=relationship_curve_x,
            y=relationship_curve_y,
            color="darkorange",
            linewidth=1.5,
        )
        bricks.append(pw.load_seaborngrid(G0, label=f"brick{i*2}"))

        #  display data after relationship removal (residue):
        G1 = sns.JointGrid(
            dropna=True, xlim=(-10, 10), ylim=(-10, 10), marginal_ticks=False
        )
        # plot datapoints before relationship removal:
        sns.scatterplot(
            ax=G1.ax_joint,
            x=res_x.cpu().detach().numpy(),
            y=residue[i].cpu().detach().numpy(),
        )
        # plot kde:
        sns.kdeplot(
            x=res_x.cpu().detach().numpy(),
            linewidth=1.25,
            ax=G1.ax_marg_x,
            bw_adjust=1.25,
            fill=True,
            common_norm=True,
        )
        sns.kdeplot(
            y=residue[i].cpu().detach().numpy(),
            linewidth=1.25,
            ax=G1.ax_marg_y,
            bw_adjust=1.25,
            fill=True,
            common_norm=True,
        )
        # plot relationships:
        sns.lineplot(
            ax=G1.ax_joint,
            x=res_x.cpu().detach().numpy(),
            y=relationship_residue[i].cpu().detach().numpy(),
            color="darkorange",
            linewidth=1.5,
        )
        bricks.append(pw.load_seaborngrid(G1, label=f"brick{i*2+1}"))

    quilt = (
        (bricks[0] | bricks[1])
        / (bricks[2] | bricks[3])
        / (bricks[4] | bricks[5])
        / (bricks[6] | bricks[7])
        / (bricks[8] | bricks[9])
        / (bricks[10] | bricks[11])
    )
    quilt.savefig("seaborn_subplots.png")

    # calculate entropy after relationship removal:
    #  final_model = Unbounded_Metalog_Model(init_a=torch.ones([channels, 7])).to(device)
    final_model = Metalog_Model(
        init_a=torch.ones([channels, 7]).to(device),
        #  lower_bounds=torch.ones([channels, 1]).to(device) * -1000,
        #  upper_bounds=torch.ones([channels, 1]).to(device) * 1000,
    ).to(device)
    Metalog_Fit_Closed_Form(final_model, residue)
    final_entropy = final_model.estimate_entropy()

    print("initial_entropy:", initial_entropy.tolist())
    print("final_entropy:", final_entropy.tolist())
    print("entropy difference:", (final_entropy - initial_entropy).tolist())


### Generate Data: ###
samples_shape = [512]

base_dist = Normal(
    torch.zeros(samples_shape).float().to(device),
    torch.ones(samples_shape).float().to(device) * 3,
)  # known skew = 0, known kurt = 0
base_var = base_dist.sample()

eA = Normal(
    torch.zeros(samples_shape).float().to(device),
    torch.ones(samples_shape).float().to(device) * 1,
)
a = base_var + eA.sample()

eB = Normal(
    torch.zeros(samples_shape).float().to(device),
    torch.ones(samples_shape).float().to(device) * 3,
)
b = eB.sample()

eC = Normal(
    torch.zeros(samples_shape).float().to(device),
    torch.ones(samples_shape).float().to(device) * 1,
)
c = (0.5 * -base_var.pow(2)) + eC.sample() + 5

eD = Normal(
    torch.zeros(samples_shape).float().to(device),
    torch.ones(samples_shape).float().to(device) * 1,
)
d = (0.5 * base_var.pow(2)) + eD.sample() - 5

eE = Normal(
    torch.zeros(samples_shape).float().to(device),
    torch.ones(samples_shape).float().to(device) * 1,
)
e = 5 * torch.sin(base_var) + eE.sample()

data = torch.stack((base_var, a, b, c, d, e), 0)

print("data.shape:", data.shape)

mutual_information(data, 0, num_terms=10)
