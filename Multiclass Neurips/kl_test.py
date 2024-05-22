import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt

device = "cuda" if t.cuda.is_available() else "cpu"
t.manual_seed(42)


class Gaussian(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mean = nn.Parameter(t.randn(d, device=device))
        self.scale = nn.Parameter(t.randn(d, device=device) / 10)

    def __hash__(self):
        return hash(self.mean.data + self.scale.data)

    @property
    def var(self):
        return t.exp(self.scale * 2)

    def kl(self, other):
        term1 = t.log(other.var / self.var).sum()
        term2 = ((self.var + (self.mean - other.mean) ** 2) / other.var).sum()
        term3 = -self.mean.numel()
        return (term1 + term2 + term3) / 2


def train(g1: Gaussian, g2: Gaussian, lr: float, epochs: int):
    optimizer = t.optim.Adam(g1.parameters(), lr=lr)
    kls = []
    scales = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}/{str(epochs)}")
        scales.append(g1.scale[0].clone().detach().cpu())
        kl = g1.kl(g2)
        kls.append(kl.detach().cpu())
        optimizer.zero_grad()
        kl.backward()
        optimizer.step()
    return kls, scales


D = 10**7
g1 = Gaussian(D)
g2 = Gaussian(D)

kls, scales = train(g1, g2, 0.01, 2000)  # You maybe want to lower the learning rate after 100 epochs with lr=0.1
plt.plot(kls)
plt.yscale("log")
plt.show()
plt.plot(scales)
plt.show()
