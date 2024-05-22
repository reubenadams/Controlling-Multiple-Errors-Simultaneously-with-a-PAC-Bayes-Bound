from dataclasses import dataclass
from typing import Optional


@dataclass
class DataConfig:
    subset_size: Optional[int] = None
    batch_size: int = 100
    prior_prop: float = 0.45
    val_prop: float = 0.05
    cert_prop: float = 0.5


@dataclass
class StochLinConfig:
    d_in: int
    d_out: int
    bias: float = 0.0  # Note the default value in DZ is 0.1 for l1 and 0.0 thereafter
    norm_mean: float = 0.0
    norm_stdv: float = 0.04
    trunc_min: float = -0.08
    trunc_max: float = 0.08
    scale: float = -6.0


@dataclass
class StochMLPConfig:
    d_in: int = 784
    d_hidden: int = 100
    d_out: int = 1
    bias_l1: float = 0.1
    bias_l2_onwards: float = 0.0
    stoch_lin_configs = [
        StochLinConfig(bias=bias_l1, d_in=d_in, d_out=d_hidden),
        StochLinConfig(bias=bias_l2_onwards, d_in=d_hidden, d_out=d_out)
    ]



@dataclass
class DetTrainConfig:
    lr: float = 0.01  # 0.01 in Dz
    momentum: float = 0.1
    epochs: int = 20  # 20 in Dz


@dataclass
class StochTrainConfig:
    lr: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 15
    b: float = 100.0
    c: float = 0.1  # 0.1 in Dz
    delta: float = 0.025
    num_grad_samples: int = 1
    prior_scale: float = -6.0
    reset_noise: bool = False


@dataclass
class FinalBoundConfig:
    bound_mc_samples: int = 1000
    delta_prime: float = 0.01


@dataclass
class MasterConfig:
    data_config: DataConfig = DataConfig()
    stoch_mlp_config: StochMLPConfig = StochMLPConfig()
    det_train_config: DetTrainConfig = DetTrainConfig()
    stoch_train_config: StochTrainConfig = StochTrainConfig()
    final_bound_config: FinalBoundConfig = FinalBoundConfig()
