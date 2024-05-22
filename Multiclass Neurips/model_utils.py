from simple_stoch_models import *
from configs import *


def create_models(config: StochMLPConfig, device):
    model_P = MLP(config.d_in, config.d_hidden, config.d_out)
    model_Q = StochMLP(config)
    model_P.to(device)
    model_Q.to(device)
    return model_P, model_Q
