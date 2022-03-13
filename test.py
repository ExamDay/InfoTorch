import torch
from tools import KernelDensityEstimate
from torch.distributions import Normal
from torch.distributions.exponential import Exponential
from torch.distributions.log_normal import LogNormal
from torch.distributions.chi2 import Chi2
from torch.distributions.uniform import Uniform

device = "cuda"

# shape of data drawn from each distribution.
samples_shape = [128, 128, 2, 2]
# point at which to bond data.
bond_point = 1
# default setup creates final dataset of shape [128, 4, 128, 2, 2]
# ^ which in this case is a batch of 128 instances of 4 different sources of random data (each
# source being a different probability distribution function) each with their own 128 samples of
# shape 2x2.

M = Normal(
    torch.zeros(samples_shape).to(device), torch.ones(samples_shape).to(device)
)  # known skew = 0, known kurt = 0
a0 = M.sample().unsqueeze(bond_point)
print("a0.shape", a0.shape)

M = LogNormal(torch.zeros(samples_shape).to(device), torch.ones(samples_shape).to(device))
# ^ known skew = 6.1848, known kurt = 110.936
a1 = M.sample().unsqueeze(bond_point)
print("a1.shape", a1.shape)

M = Chi2(torch.ones(samples_shape).to(device) * 4)  # known skew = 1.4142, known kurt = 3
a2 = M.sample().unsqueeze(bond_point)
print("a2.shape", a2.shape)

M = Exponential(torch.ones(samples_shape).to(device))  # known skew = 2, known kurt = 6
a3 = M.sample().unsqueeze(bond_point)
print("a3.shape", a3.shape)

import matplotlib.pyplot as plt

plt.style.use("seaborn-whitegrid")
import numpy as np

fig = plt.figure()
ax = plt.axes()

data = torch.cat((a0, a1, a2, a3), bond_point)

steps = 256  # steps for estimation
# ^ this must be an integer
a = -5  # lower integration bound
b = 10  # upper integration bound
x_tics = torch.Tensor(np.linspace(a, b, steps)).to(device)
print("\n\nx_tics.shape:", x_tics.shape)

y_tics = KernelDensityEstimate(
    data,
    x_tics=x_tics,
    bandwidth_adjustment=0.25,
    dim=2,
)
# OPTIONAL: average density estimates across all batches to get a single estimate of higher
# confidence
y_tics = torch.mean(y_tics, dim=0)
print("\n\ny_tics.shape:", y_tics.shape)

eps = 1e-5

# NORMAL:
ground_truth_y_tics = Normal(loc=0, scale=1).log_prob(x_tics).exp().cpu().detach()
kde_area = torch.trapz(y=y_tics[0], x=x_tics)
ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "r")
ax.plot(x_tics.cpu().detach().numpy(), y_tics[0].cpu().detach().numpy(), "r--")
print("kde_area0:", kde_area)
overlap_fn = torch.amin(
    torch.stack((y_tics[0].to(device), ground_truth_y_tics.to(device))), 0
)
overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
gabe_distance = 1 - overlap_area
print("gabe_distance0:", gabe_distance)

# LOG NORMAL:
ground_truth_y_tics = (
    LogNormal(0, 1).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
)
kde_area = torch.trapz(y=y_tics[1], x=x_tics)
ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "m")
ax.plot(x_tics.cpu().detach().numpy(), y_tics[1].cpu().detach().numpy(), "m--")
print("kde_area1:", kde_area)
overlap_fn = torch.amin(
    torch.stack((y_tics[1].to(device), ground_truth_y_tics.to(device))), 0
)
overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
gabe_distance = 1 - overlap_area
print("gabe_distance1:", gabe_distance)

# CHI2:
ground_truth_y_tics = Chi2(4).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
kde_area = torch.trapz(y=y_tics[2], x=x_tics)
ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "g")
ax.plot(x_tics.cpu().detach().numpy(), y_tics[2].cpu().detach().numpy(), "g--")
print("kde_area2:", kde_area)
overlap_fn = torch.amin(
    torch.stack((y_tics[2].to(device), ground_truth_y_tics.to(device))), 0
)
overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
gabe_distance = 1 - overlap_area
print("gabe_distance2:", gabe_distance)

# EXPONENTIAL:
ground_truth_y_tics = (
    Exponential(1).log_prob(x_tics.clamp(min=eps)).exp().cpu().detach()
)
gty_max = ground_truth_y_tics.max()
for idx, item in enumerate(ground_truth_y_tics.unsqueeze(1)):
    if item == gty_max:
        ground_truth_y_tics[idx] = 0

kde_area = torch.trapz(y=y_tics[3], x=x_tics)
ax.plot(x_tics.cpu().detach().numpy(), ground_truth_y_tics, "b")
ax.plot(x_tics.cpu().detach().numpy(), y_tics[3].cpu().detach().numpy(), "b--")
print("kde_area3:", kde_area)
overlap_fn = torch.amin(
    torch.stack((y_tics[3].to(device), ground_truth_y_tics.to(device))), 0
)
overlap_area = torch.trapz(y=overlap_fn, x=x_tics)
gabe_distance = 1 - overlap_area
print("gabe_distance3:", gabe_distance)

ax.legend(
    [
        "standard normal",
        "kde normal",
        "standard log-normal",
        "kde log-normal",
        "standard chi2",
        "kde chi2",
        "standard exponential",
        "kde exponential",
    ]
)
plt.show()
