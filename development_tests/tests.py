import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.exponential import Exponential
from torch.distributions.log_normal import LogNormal
from torch.distributions.chi2 import Chi2
from torch.distributions.uniform import Uniform

device = "cuda"

#  # shape of data drawn from each distribution.
#  samples_shape = [8, 2048]
#  # point at which to bond data.
#  bond_point = 1
#  # default setup creates final dataset of shape [128, 4, 128, 2, 2]
#  # ^ which in this case is a batch of 128 instances of 4 different sources of random data (each
#  # source being a different probability distribution function) each with their own 128 samples of
#  # shape 2x2.

#  M = Normal(
#      torch.zeros([samples_shape[0], samples_shape[1] // 2]).to(device) - 1,
#      torch.ones([samples_shape[0], samples_shape[1] // 2]).to(device),
#  )  # known skew = 0, known kurt = 0
#  mix = M.sample()
#  M = Normal(
#      torch.zeros([samples_shape[0], samples_shape[1] // 2]).to(device) + 2,
#      torch.ones([samples_shape[0], samples_shape[1] // 2]).to(device) * 4,
#  )  # known skew = 0, known kurt = 0
#  norm_data = torch.cat((mix.unsqueeze(1), M.sample().unsqueeze(1)), 1)
#  #  print("norm_data.shape", norm_data.shape)

#  M = Normal(
#      torch.zeros(samples_shape).to(device), torch.ones(samples_shape).to(device)
#  )  # known skew = 0, known kurt = 0
#  a0 = M.sample().unsqueeze(bond_point)
#  #  print("a0.shape", a0.shape)

#  M = LogNormal(
#      torch.zeros(samples_shape).to(device), torch.ones(samples_shape).to(device)
#  )
#  # ^ known skew = 6.1848, known kurt = 110.936
#  a1 = M.sample().unsqueeze(bond_point)
#  #  print("a1.shape", a1.shape)

#  M = Chi2(
#      torch.ones(samples_shape).to(device) * 4
#  )  # known skew = 1.4142, known kurt = 3
#  a2 = M.sample().unsqueeze(bond_point)
#  #  print("a2.shape", a2.shape)

#  M = Exponential(torch.ones(samples_shape).to(device))  # known skew = 2, known kurt = 6
#  a3 = M.sample().unsqueeze(bond_point)
#  #  print("a3.shape", a3.shape)

#  data = torch.cat((a0, a1, a2, a3), bond_point)

###################################################################################################
### Estimation: ###
class Normal_Model(nn.Module):
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
        """method to run on "constrain" step of training.
        Easiest method for optimization under constraint is Projection Optimization by simply
        clamping parameters to bounds after each update. This is certainly not the most efficent way
        to optimize under constraints, but it is stable and it works. Good enough for now."""
        #  can't have negative standard deviation so lets prevent that.
        eps = 1e-6
        self.std.data = model.std.data.clamp(min=eps)

    def log_prob(self, x):
        return (
            -torch.log(self.std.unsqueeze(-1))
            - (self.ln2p / 2)
            - ((x - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)).pow(2) / 2
        )

    def forward(self, x):
        return self.log_prob(x).exp()


def MLE_Fit(model, data, dim=1, lr=5e-2, iters=250):
    """Fits the parameters of the model to the data. Provided model must have implimented log_prob()
    and constrain() methods."""
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    #  print("model parameters:", [x for x in model.parameters()])
    #  data = data.flatten(dim)
    for i in range(iters):
        nll = -torch.sum(model.log_prob(data))
        nll.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.constrain()


class Unbounded_Metalog_Model(nn.Module):
    '''
    An implimentation of unbounded metalog models.
    '''
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

        ### Define basis functions for derivative of quantile function in terms of cumulative
        ### probability. (^ derivative of quantile function):
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

        # Start derivative quantile basis functions:
        self.dqf_basis_functions = [dqg1, dqg2, dqg3, dqg4]
        # Additional dqf basis functions as needed:
        self.dqf_basis_functions = self.dqf_basis_functions + [
            dqgj_odd if x % 2 == 0 else dqgj_even for x in range(self.n - 4)
        ]
        # Trim as needed:
        self.dqf_basis_functions = self.dqf_basis_functions[: self.n]

    def constrain(self):
        '''Coefficients are unconstrained in this case.'''
        pass

    def quantile(self, y):
        '''
        Quantile of cumulative probability "y".  (returns x-position of cumulative probability "y".
        This is an inverse CDF)
        '''
        x_values = sum(
            [
                self.a[:, idx].unsqueeze(-1) * f(y, idx)
                for idx, f in enumerate(self.qf_basis_functions)
            ]
        )
        return x_values

    def derivative_quantile(self, y):
        '''
        Derivative of quantile as function of cumulative probability "y".
        (AKA: quantile density function.)
        '''
        return sum(
            [
                self.a[:, idx].unsqueeze(-1) * f(y, idx)
                for idx, f in enumerate(self.dqf_basis_functions)
            ]
        )

    def prob_ito_cumprob(self, y):
        '''Probability density in terms of cumulative probability "y".'''
        return self.derivative_quantile(y).pow(-1)

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
        '''Approximates log of probability density at a batch of tensors "x".'''
        return torch.log(self.prob(x))

    def estimate_entropy(self, steps=256):
        '''Estimates shannon entropy of the distribution in nats by numeric integration.'''
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
        '''Simulates data of shape "shape" by inverse tranform sampling.'''
        eps = 1e-7
        return self.quantile(torch.rand(shape).clamp(min=eps, max=1-eps).to(self.a.device))

    def forward(self, x):
        '''
        By default: Approximates probability density at a batch of tensors "x" by asymptotically
        bounded approach. There is currently no known closed-form inverse metalog.
        '''
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


def Metalog_Fit(model, data, dim=1, lr=5e-2, iters=250):
    """Fits the parameters of the model to the data. Provided model must have implimented log_prob()
    and constrain() methods."""
    ecdf = ECDF(data, dim=1, reach_limits=False)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    for i in range(iters):
        x, y = ecdf.split(1, dim=dim)
        loss = F.mse_loss(y, model.cumulative(x))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.constrain()


def Metalog_Fit_Closed_Form(model, data):
    """Fits the parameters of the model to the data by a closed-form linear least-squares method."""
    ecdf = ECDF(data, dim=1, reach_limits=False)
    print("ecdf[0].shape:", ecdf[0].shape)
    print("ecdf[1].shape:", ecdf[1].shape)
    x, y = ecdf
    x = x.float()
    print("x:", x)
    print("x.shape:", x.shape)
    y = y.float()
    print("y:", y)
    print("y.shape:", y.shape)

    Y_cols = [f(y, idx) for idx, f in enumerate(model.qf_basis_functions)]
    print("Y_cols:", Y_cols)
    Y = torch.stack(Y_cols, -1)
    #  print("Y:", Y)
    print("Y.shape:", Y.shape)
    a = torch.bmm(
        torch.linalg.solve(torch.bmm(Y.transpose(1, 2), Y), Y.transpose(1, 2)),
        x.unsqueeze(-1),
    ).flatten(1)
    #  a = torch.bmm(
    #      torch.bmm(torch.inverse(torch.bmm(Y.transpose(1, 2), Y)), Y.transpose(1, 2)),
    #      x.unsqueeze(-1),
    #  ).flatten(1)
    print("a", a)
    print("a.shape", a.shape)
    model.a.data = a


samples_shape = [64]
Ma = Normal(
    torch.zeros(samples_shape).to(device) - 1,
    torch.ones(samples_shape).to(device),
)  # known skew = 0, known kurt = 0
da = Ma.sample()
print("da.shape:", da.shape)

Mb = Normal(
    torch.zeros(samples_shape).to(device) + 2,
    torch.ones(samples_shape).to(device) * 4,
)  # known skew = 0, known kurt = 0
db = Mb.sample()
print("db.shape:", db.shape)

mix = torch.cat((da[:samples_shape[0]//2], db[:samples_shape[0]//2]), 0)
print("mix.shape:", mix.shape)

norm_data = torch.stack((da, db, mix), 0)
print("norm_data.shape", norm_data.shape)

model = Normal_Model(
    init_mean=torch.zeros([norm_data.shape[0]]),
    init_std=torch.ones([norm_data.shape[0]]),
).to(device)
MLE_Fit(model, norm_data, dim=1, iters=250)

metalog_model = Unbounded_Metalog_Model(init_a=torch.zeros([2, 7])).to(device)
Metalog_Fit_Closed_Form(metalog_model, norm_data)


norm_ecdf = ECDF(norm_data, dim=1)
print("norm_ecdf[0].shape: ", norm_ecdf[0].shape)
print("norm_ecdf[1].shape: ", norm_ecdf[1].shape)

###################################################################################################

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
import numpy as np

fig = plt.figure()
ax = plt.axes()

steps = 16384  # steps for estimation
# ^ this must be an integer
a = -10  # lower integration bound
b = 15  # upper integration bound
x_tics = torch.Tensor(np.linspace(a, b, steps)).to(device)
print("\n\nx_tics.shape:", x_tics.shape)

y_tics = model(x_tics.repeat(norm_data.shape[0], 1))
print("\n\ny_tics.shape:", y_tics.shape)

eps = 1e-5

# Ground Truth:
ground_truth_y_tics1 = Normal(loc=-1, scale=1).log_prob(x_tics).exp().cpu().detach()
ground_truth_y_tics2 = Normal(loc=2, scale=4).log_prob(x_tics).exp().cpu().detach()
ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics1.cpu().detach().numpy(), "k")
ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics2.cpu().detach().numpy(), "k")

# Normal Model MLE Fit:
ax.plot(x_tics.cpu().detach().numpy(), y_tics[0].cpu().detach().numpy(), "r--")
ax.plot(x_tics.cpu().detach().numpy(), y_tics[1].cpu().detach().numpy(), "r--")

# ECDF:
ecdfX1 = norm_ecdf[0][0, :]
print("ecdfX1:", ecdfX1)
ecdfY1 = norm_ecdf[1][0, :]
print("ecdfY1:", ecdfY1)

ecdfX2 = norm_ecdf[0][1, :]
print("ecdfX2:", ecdfX2)
ecdfY2 = norm_ecdf[1][1, :]
print("ecdfY2:", ecdfY2)

ax.plot(ecdfX1.cpu().detach().numpy(), ecdfY1.cpu().detach().numpy(), "g")
ax.plot(ecdfX2.cpu().detach().numpy(), ecdfY2.cpu().detach().numpy(), "g")

# Normal Histograms
ax.hist(norm_data[0, :].cpu().detach().numpy(), bins=16, density=True)
ax.hist(norm_data[1, :].cpu().detach().numpy(), bins=16, density=True)

#  # LOG NORMAL:
#  ground_truth_y_tics = (
#      LogNormal(0, 1).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
#  )
#  ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "m")
#  ax.plot(x_tics.cpu().detach().numpy(), y_tics[1].cpu().detach().numpy(), "m--")

#  # CHI2:
#  ground_truth_y_tics = Chi2(4).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
#  ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "g")
#  ax.plot(x_tics.cpu().detach().numpy(), y_tics[2].cpu().detach().numpy(), "g--")

#  # EXPONENTIAL:
#  ground_truth_y_tics = (
#      Exponential(1).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
#  )
#  gty_max = ground_truth_y_tics.max()
#  for idx, item in enumerate(ground_truth_y_tics.unsqueeze(1)):
#      if item == gty_max:
#          ground_truth_y_tics[idx] = 0 #  ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "b") #  ax.plot(x_tics.cpu().detach().numpy(), y_tics[3].cpu().detach().numpy(), "b--")

ax.legend(
    [
        "Ground Truth Normal 1",
        "Ground Truth Normal 2",
        "MLE Normal 1",
        "MLE Normal 2",
        "ECDF Normal 1",
        "ECDF Normal 2",
        "Histogram Normal 1",
        "Histogram Normal 2",
        #  "standard log-normal",
        #  "MLE log-normal",
        #  "standard chi2",
        #  "MLE chi2",
        #  "standard exponential",
        #  "MLE exponential",
    ]
)
plt.show()

###################################################################################################
### Metalog Modeling ###
###################################################################################################

print("\n\n### Metalog Modeling ###:\n\n")

fig = plt.figure()
ax = plt.axes()

# ECDF:

ecdfX1 = norm_ecdf[0][0, :]
ecdfY1 = norm_ecdf[1][0, :]

ecdfX2 = norm_ecdf[0][1, :]
ecdfY2 = norm_ecdf[1][1, :]

ecdfX3 = norm_ecdf[0][2, :]
ecdfY3 = norm_ecdf[1][2, :]

ax.plot(ecdfX1.cpu().detach().numpy(), ecdfY1.cpu().detach().numpy(), "k")
ax.plot(ecdfX2.cpu().detach().numpy(), ecdfY2.cpu().detach().numpy(), "k")
ax.plot(ecdfX3.cpu().detach().numpy(), ecdfY3.cpu().detach().numpy(), "k")

# Metalog CDF:
steps = 16384  # steps for estimation
# ^ this must be an integer
a = 1e-6  # lower integration bound
b = 1 - 1e-6  # upper integration bound
y_tics = torch.Tensor(np.linspace(a, b, steps)).to(device)
print("\n\ny_tics.shape:", y_tics.shape)

x_tics = metalog_model.quantile(y_tics)
print("\n\nx_tics.shape:", x_tics.shape)

eps = 1e-5

ax.plot(x_tics[0].cpu().detach().numpy(), y_tics.cpu().detach().numpy(), "r--")
ax.plot(x_tics[1].cpu().detach().numpy(), y_tics.cpu().detach().numpy(), "r--")
ax.plot(x_tics[2].cpu().detach().numpy(), y_tics.cpu().detach().numpy(), "r--")

ax.legend(
    [
        "Ground Truth ECDF Normal 1",
        "Ground Truth ECDF Normal 2",
        "Ground Truth ECDF Mix",
        "Metalog CDF Normal 1",
        "Metalog CDF Normal 2",
        "Metalog CDF Mix",
    ]
)

plt.show()

print("\n\nPROB:\n\n")
fig = plt.figure()
ax = plt.axes()

steps = 16384  # steps for estimation
# ^ this must be an integer
a = -10  # lower integration bound
b = 15  # upper integration bound
x_tics = torch.Tensor(np.linspace(a, b, steps)).to(device)

# Ground Truth PDF:
ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics1.cpu().detach().numpy(), "k")
ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics2.cpu().detach().numpy(), "k")
Ma = Normal(
    torch.zeros([steps]).to(device) - 1,
    torch.ones([steps]).to(device),
)  # known skew = 0, known kurt = 0
Mb = Normal(
    torch.zeros([steps]).to(device) + 2,
    torch.ones([steps]).to(device) * 4,
)  # known skew = 0, known kurt = 0
mix = (Ma.log_prob(x_tics).exp() + Mb.log_prob(x_tics).exp()) / 2
ax.plot(x_tics.cpu().detach().numpy(), mix.cpu().detach().numpy(), "k")

# Metalog PDF:
steps = 64  # steps for estimation
# ^ this must be an integer
a = -10  # lower integration bound
b = 15  # upper integration bound
x_tics = torch.Tensor(np.linspace(a, b, steps)).to(device)
y_tics = metalog_model.prob(x_tics.repeat(norm_data.shape[0], 1))

eps = 1e-5
ax.plot(x_tics.cpu().detach().numpy(), y_tics[0].cpu().detach().numpy(), "r--")
ax.plot(x_tics.cpu().detach().numpy(), y_tics[1].cpu().detach().numpy(), "r--")
ax.plot(x_tics.cpu().detach().numpy(), y_tics[2].cpu().detach().numpy(), "r--")

print("entropy:", metalog_model.estimate_entropy(steps=4096))

ax.hist(norm_data[0, :].cpu().detach().numpy(), bins=16, density=True)
ax.hist(norm_data[1, :].cpu().detach().numpy(), bins=16, density=True)
ax.hist(norm_data[2, :].cpu().detach().numpy(), bins=32, density=True)

ax.legend(
    [
        "Ground Truth Normal PDF 1",
        "Ground Truth Normal PDF 2",
        "Ground Truth Mix PDF",
        "Metalog Normal PDF 1",
        "Metalog Normal PDF 2",
        "Metalog Mix PDF",
        "Histogram Normal 1",
        "Histogram Normal 2",
        "Histogram Mix",
    ]
)

plt.show()

#  # LOG NORMAL:
#  ground_truth_y_tics = (
#      LogNormal(0, 1).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
#  )
#  ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "m")
#  ax.plot(x_tics.cpu().detach().numpy(), y_tics[1].cpu().detach().numpy(), "m--")

#  # CHI2:
#  ground_truth_y_tics = Chi2(4).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
#  ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "g")
#  ax.plot(x_tics.cpu().detach().numpy(), y_tics[2].cpu().detach().numpy(), "g--")

#  # EXPONENTIAL:
#  ground_truth_y_tics = (
#      Exponential(1).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
#  )
#  gty_max = ground_truth_y_tics.max()
#  for idx, item in enumerate(ground_truth_y_tics.unsqueeze(1)):
#      if item == gty_max:
#          ground_truth_y_tics[idx] = 0

#  ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "b")
#  ax.plot(x_tics.cpu().detach().numpy(), y_tics[3].cpu().detach().numpy(), "b--")

#  ax.legend(
#      [
#          "Ground Truth ECDF Normal 1",
#          "Ground Truth ECDF Normal 2",
#          "Ground Truth ECDF Mix",
#          "Metalog CDF Normal 1",
#          "Metalog CDF Normal 2",
#          "Metalog CDF Mix",
#          "Ground Truth Normal PDF 1",
#          "Ground Truth Normal PDF 2",
#          "Ground Truth Mix PDF",
#          "Metalog Normal PDF 1",
#          "Metalog Normal PDF 2",
#          "Metalog Mix PDF",
#          "Histogram Normal 1",
#          "Histogram Normal 2",
#          "Histogram Mix",
#          #  "standard log-normal",
#          #  "MLE log-normal",
#          #  "standard chi2",
#          #  "MLE chi2",
#          #  "standard exponential",
#          #  "MLE exponential",
#      ]
#  )
#  plt.show()
