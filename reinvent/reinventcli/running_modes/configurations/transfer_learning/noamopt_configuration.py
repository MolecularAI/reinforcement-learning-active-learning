from dataclasses import dataclass

@dataclass
class NoamoptConfiguration:
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.98
    eps: float = 1e-9
    factor: float = 1.0
    warmup: float = 4000
