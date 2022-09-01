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
        self.std.data = self.std.data.clamp(min=eps)

    def log_prob(self, x):
        return (
            -torch.log(self.std.unsqueeze(-1))
            - (self.ln2p / 2)
            - ((x - self.mean.unsqueeze(-1)) / self.std.unsqueeze(-1)).pow(2) / 2
        )

    def entropy(self, steps=256):
        """Returns shannon entropy of the distribution in nats."""
        return 0.5 * torch.log(2 * torch.pi * self.std.pow(2)) + 0.5

    def forward(self, x):
        return self.log_prob(x).exp()


def MLE_Fit(model, data, dim=1, lr=5e-2, iters=1000):
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

    initial_model = Normal_Model(
        init_mean=torch.zeros([channels]).to(device),
        init_std=torch.ones([channels]).to(device),
    ).to(device)
    MLE_Fit(initial_model, data)
    initial_entropy = initial_model.entropy()

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
    #  bricks = []
    #  for i in range(channels):
    #      #  fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

    #      G0 = sns.JointGrid(
    #          dropna=True, xlim=(-10, 10), ylim=(-10, 10), marginal_ticks=False
    #      )

    #      # plot datapoints before relationship removal:
    #      sns.scatterplot(
    #          ax=G0.ax_joint,
    #          x=data[idx].cpu().detach().numpy(),
    #          y=data[i].cpu().detach().numpy(),
    #      )
    #      # plot kde:
    #      sns.kdeplot(
    #          x=data[idx].cpu().detach().numpy(),
    #          linewidth=1.5,
    #          ax=G0.ax_marg_x,
    #          bw_adjust=1.25,
    #          fill=True,
    #          common_norm=True,
    #      )
    #      sns.kdeplot(
    #          y=data[i].cpu().detach().numpy(),
    #          linewidth=1.5,
    #          ax=G0.ax_marg_y,
    #          bw_adjust=1.25,
    #          fill=True,
    #          common_norm=True,
    #      )
    #      # plot relationships:
    #      relationship_curve_x = torch.DoubleTensor(
    #          np.linspace(data[idx].min().cpu(), data[idx].max().cpu(), 2048)
    #      )
    #      relationship_curve_y = sum(
    #          [
    #              coeff * relationship_curve_x.pow(idx)
    #              for idx, coeff in enumerate(a[i].cpu())
    #          ]
    #      )

    #      sns.lineplot(
    #          ax=G0.ax_joint,
    #          #  x=x.cpu().detach().numpy(),
    #          #  y=y[i].cpu().detach().numpy(),
    #          x=relationship_curve_x,
    #          y=relationship_curve_y,
    #          color="darkorange",
    #          linewidth=1.5,
    #      )
    #      bricks.append(pw.load_seaborngrid(G0, label=f"brick{i*2}"))

    #      #  display data after relationship removal (residue):
    #      G1 = sns.JointGrid(
    #          dropna=True, xlim=(-10, 10), ylim=(-10, 10), marginal_ticks=False
    #      )
    #      # plot datapoints before relationship removal:
    #      sns.scatterplot(
    #          ax=G1.ax_joint,
    #          x=res_x.cpu().detach().numpy(),
    #          y=residue[i].cpu().detach().numpy(),
    #      )
    #      # plot kde:
    #      sns.kdeplot(
    #          x=res_x.cpu().detach().numpy(),
    #          linewidth=1.25,
    #          ax=G1.ax_marg_x,
    #          bw_adjust=1.25,
    #          fill=True,
    #          common_norm=True,
    #      )
    #      sns.kdeplot(
    #          y=residue[i].cpu().detach().numpy(),
    #          linewidth=1.25,
    #          ax=G1.ax_marg_y,
    #          bw_adjust=1.25,
    #          fill=True,
    #          common_norm=True,
    #      )
    #      # plot relationships:
    #      sns.lineplot(
    #          ax=G1.ax_joint,
    #          x=res_x.cpu().detach().numpy(),
    #          y=relationship_residue[i].cpu().detach().numpy(),
    #          color="darkorange",
    #          linewidth=1.5,
    #      )
    #      bricks.append(pw.load_seaborngrid(G1, label=f"brick{i*2+1}"))

    #  quilt = (
    #      (bricks[0] | bricks[1])
    #      / (bricks[2] | bricks[3])
    #      / (bricks[4] | bricks[5])
    #      / (bricks[6] | bricks[7])
    #      / (bricks[8] | bricks[9])
    #      / (bricks[10] | bricks[11])
    #  )
    #  quilt.savefig("seaborn_subplots.png")

    # calculate entropy after relationship removal:
    final_model = Normal_Model(
        init_mean=torch.zeros([channels]).to(device),
        init_std=torch.ones([channels]).to(device),
    ).to(device)
    MLE_Fit(final_model, residue)
    final_entropy = final_model.entropy()

    print("initial_entropy:", initial_entropy.tolist())
    print("final_entropy:", final_entropy.tolist())
    print("entropy difference:", (final_entropy - initial_entropy).tolist())


### Generate Data: ###
samples_shape = [2**16]

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
