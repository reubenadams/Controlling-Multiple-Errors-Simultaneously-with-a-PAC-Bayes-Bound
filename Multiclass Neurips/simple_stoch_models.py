from pac_bayes.model_utils import *
from pac_bayes.kl_utils import *
from configs import StochLinConfig, StochMLPConfig
import torch.nn as nn
import torch.nn.functional as F


def stdv(scale):
    return t.exp(scale)


def var(scale):
    return t.exp(2 * scale)


class StochLin(nn.Module):
    def __init__(
        self,
        config: StochLinConfig,
    ):
        super().__init__()
        self.stoch = True  # Starts of stochastic
        self.weight_mean = nn.Parameter(t.empty(config.d_out, config.d_in))
        nn.init.trunc_normal_(
            self.weight_mean,
            config.norm_mean,
            config.norm_stdv,
            config.trunc_min,
            config.trunc_max,
        )
        self.weight_scale = nn.Parameter(t.full((config.d_out, config.d_in), config.scale))
        self.bias_mean = nn.Parameter(t.full((config.d_out,), config.bias))
        self.bias_scale = nn.Parameter(t.full((config.d_out,), config.scale))

    @property
    def weight_var(self):
        return var(self.weight_scale)

    @property
    def bias_var(self):
        return var(self.bias_scale)

    # TODO: Is this the correct way to copy? Got it from here: https://discuss.pytorch.org/t/copy-weights-only-from-a-networks-parameters/5841
    def load_mean(self, other: nn.Linear):
        self.weight_mean.data.copy_(other.weight.data)
        self.bias_mean.data.copy_(other.bias.data)
        # self.weight_mean.data = other.weight_mean.data.clone()
        # self.bias_mean.data = other.bias_mean.data.clone()

    # TODO: Check. Do you need a .detach()?
    def set_scale_from_mean(self):
        self.weight_scale.data = 0.5 * t.log(t.abs(self.weight_mean.data.clone()))
        self.bias_scale.data = 0.5 * t.log(t.abs(self.bias_mean.data.clone()))

    def weight_noise(self):
        # t.manual_seed(0)
        return t.randn(self.weight_mean.shape, device=self.weight_mean.device)

    def bias_noise(self):
        # t.manual_seed(0)
        return t.randn(self.bias_mean.shape, device=self.bias_mean.device)

    def forward(self, x):
        if self.stoch:
            return self.stoch_forward(x)
        return self.det_forward(x)

    def det_forward(self, x):
        return F.linear(x, self.weight_mean, self.bias_mean)

    def stoch_forward(self, x):
        stoch_weight = self.weight_mean + self.weight_noise() * stdv(self.weight_scale)
        stoch_bias = self.bias_mean + self.bias_noise() * stdv(self.bias_scale)
        return F.linear(x, stoch_weight, stoch_bias)

    def kl_divergence(self, other: nn.Linear, prior_scale: float):
        # prior_weight_scale = t.full(other.weight.shape, prior_scale)
        # prior_bias_scale = t.full(other.bias.shape, prior_scale)
        weight_kl = kl_divergence_scale(
            self.weight_mean, self.weight_scale, other.weight, prior_scale
        )
        bias_kl = kl_divergence_scale(
            self.bias_mean, self.bias_scale, other.bias, prior_scale
        )
        return weight_kl + bias_kl


class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.layer1 = nn.Linear(d_in, d_hidden)
        self.act = nn.ReLU()
        self.layer2 = nn.Linear(d_hidden, d_out)
        self.soft_layer = nn.Softmax(dim=-1)
        self.softmax_in_forward = False

    def forward(self, x):
        y_hat = self.layer2(self.act(self.layer1(x.flatten(1))))
        if self.softmax_in_forward:
            y_hat = self.soft_layer(y_hat)
        return y_hat


class StochMLP(nn.Module):
    def __init__(
        self,
        config: StochMLPConfig,
    ):
        super().__init__()
        self.layer1 = StochLin(config.stoch_lin_configs[0])
        self.act = nn.ReLU()
        self.layer2 = StochLin(config.stoch_lin_configs[1])
        self.soft_layer = nn.Softmax(dim=-1)
        self.softmax_in_forward = False

    def all_det(self):
        self.layer1.stoch = False
        self.layer2.stoch = False

    def all_stoch(self):
        self.layer1.stoch = True
        self.layer2.stoch = True

    def load_mean(self, other: MLP):
        self.layer1.load_mean(other.layer1)
        self.layer2.load_mean(other.layer2)

    def set_scale_from_mean(self):
        self.layer1.set_scale_from_mean()
        self.layer2.set_scale_from_mean()

    def forward(self, x):
        y_hat = self.layer2(self.act(self.layer1(x.flatten(1))))
        if self.softmax_in_forward:
            y_hat = self.soft_layer(y_hat)
        return y_hat

    def kl_divergence(self, other: MLP, prior_scale: float):
        kl1 = self.layer1.kl_divergence(other.layer1, prior_scale)
        kl2 = self.layer2.kl_divergence(other.layer2, prior_scale)
        return kl1 + kl2

    # TODO: Parallelise
    def mc_loss(self, x, y, loss_fn, num_mc_samples, reset_noise=False):
        """Returns MC estimate of *sum* of model losses on batch x"""
        self.all_stoch()
        losses = []
        for _ in range(num_mc_samples):
            if reset_noise:
                t.manual_seed(0)
            losses.append(loss_fn(self(x), y))
        # losses = [loss_fn(self(x), y) for _ in range(num_mc_samples)]
        return sum(losses) / num_mc_samples
