import torch
import torch.nn as nn
from .layers import RadialMuntzEdge, RadialAngularMuntzEdge, MultiCenterRadialEdge
from .utils import compute_radius

class RMNDirect(nn.Module):
    def __init__(self, d_in: int, K: int = 12, include_log: bool = True, mu_min: float = -2.0, mu_max: float = 4.0):
        super().__init__()
        self.edge = RadialMuntzEdge(K=K, include_log=include_log, mu_min=mu_min, mu_max=mu_max)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = compute_radius(x)
        return self.edge(r).unsqueeze(-1) + self.bias

    def forward_with_grad_lap(self, x: torch.Tensor):
        f, grad_f, lap_f = self.edge.forward_with_grad_lap(x, d=x.shape[-1])
        return f.unsqueeze(-1) + self.bias, grad_f, lap_f

    def exponent_separation_regularizer(self, min_sep: float = 0.1) -> torch.Tensor:
        mu = self.edge.exponents()
        gaps = mu[1:] - mu[:-1]
        return torch.relu(min_sep - gaps).mean()

class RMNAngular(nn.Module):
    def __init__(self, d_in: int, K_r: int = 6, K_a: int = 4, L_max: int = 2,
                 include_log: bool = True, mu_min: float = -2.0, mu_max: float = 4.0):
        super().__init__()
        self.edge = RadialAngularMuntzEdge(K_r=K_r, K_a=K_a, L_max=L_max, dim=d_in,
                                          include_log=include_log, mu_min=mu_min, mu_max=mu_max)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge(x).unsqueeze(-1) + self.bias

class RMNMultiCenter(nn.Module):
    def __init__(self, d_in: int, J: int = 2, K: int = 6, learn_centers: bool = True,
                 include_log: bool = True, mu_min: float = -2.0, mu_max: float = 4.0):
        super().__init__()
        self.edge = MultiCenterRadialEdge(J=J, K=K, d=d_in, learn_centers=learn_centers,
                                          include_log=include_log, mu_min=mu_min, mu_max=mu_max)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.edge(x).unsqueeze(-1) + self.bias
