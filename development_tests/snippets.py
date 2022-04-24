import torch
import numpy as np
from torch.autograd import Variable

# Draw random samples
sample = np.random.randint(10, size=(10, 1))

# Convert to Pytorch tensor
x = Variable(torch.from_numpy(sample)).type(torch.FloatTensor)

# Initialize Parameters
p = Variable(torch.rand(1), requires_grad=True)

# Initialize optimizer
optimizer = torch.optim.SGD([p], lr=0.0002)

for t in range(1500):
    optimizer.zero_grad()

    # Negative log likelihood of exponential NLL = -log (L | X_1, ..., X_n)
    NLL = -(x.size()[0] * torch.log(p) - p * torch.sum(x))

    NLL.backward()
    print("NLL", NLL, "p", p)

    optimizer.step()

print("Theoretical estimation", 1 / np.mean(sample))

self.mean = nn.Parameter(mean, requires_grad=False)
self.std = nn.Parameter(std, requires_grad=False)

a = [ 1.4479768975701366 / (0.02043 * 11), 0.4840444254564448 / (0.1364 * 11), 0.12922755927661228 / (0.1274 * 11), 0.010161105367512123 / (0.04302 * 11), 1.0376804846215344 / (0.1147 * 11), 0.23591927263384047 / (0.07915 * 11), 0.44683944200467857 / (0.0361 * 11), 88.98917833625146 / (0.1475 * 11), 121.03009245247779 / (0.03946 * 11), 0.13281818558935302 / (0.1618 * 11), 0.003026247022400305 / (0.0873 * 11), ]

a = [
    6.44 / (0.0383 * 11),  #  bce
    0.323 / (0.1045 * 11),  # vgg1
    0.0922 / (0.09886 * 11),  # vgg2
    0.0215 / (0.07672 * 11),  # effnet1
    0.822 / (0.1018 * 11),  # effnet2
    0.271 / (0.09467 * 11),  # mobilenet1
    1.13 / (0.07876 * 11),  # mobilenet2
    54.8 / (0.1164 * 11),  # densenet1
    279 / (0.08676 * 11),  # densenet2
    0.0746 / (0.1039 * 11),  # squeezenet1
    0.00315 / (0.0982 * 11),  # squeezenet
]
