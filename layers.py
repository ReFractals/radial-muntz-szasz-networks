import torch
import torch.nn as nn

from .parameterizations import make_ordered_bounded_signed, make_bounded_signed
from .utils import compute_radius, safe_pow, log_primitive, spherical_harmonics_2d, spherical_harmonics_3d, num_spherical_harmonics

class RadialMuntzEdge(nn.Module):
    """sum_k a_k r^{mu_k} + c_log * psi_log(r, mu_log)."""

    def __init__(self, K: int = 12, include_log: bool = True, init_scale: float = 0.02,
                 mu_min: float = -2.0, mu_max: float = 4.0, use_smooth_ordering: bool = True):
        super().__init__()
        self.K = K
        self.mu_min, self.mu_max = mu_min, mu_max
        self.include_log = include_log
        self.use_smooth_ordering = use_smooth_ordering

        self.a = nn.Parameter(init_scale * torch.randn(K))
        self.raw_mu = nn.Parameter(torch.randn(K))

        if include_log:
            self.c_log = nn.Parameter(torch.tensor(0.0))
            self.raw_mu_log = nn.Parameter(torch.tensor(0.0))

    def exponents(self) -> torch.Tensor:
        if self.use_smooth_ordering:
            return make_ordered_bounded_signed(self.raw_mu, self.mu_min, self.mu_max)
        return make_bounded_signed(self.raw_mu, self.mu_min, self.mu_max)

    def log_exponent(self) -> torch.Tensor | None:
        if not self.include_log:
            return None
        return 0.5 * torch.tanh(self.raw_mu_log)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        mu = self.exponents()
        out = safe_pow(r, mu) @ self.a
        if self.include_log:
            out = out + self.c_log * log_primitive(r, self.log_exponent())
        return out

    def forward_with_grad_lap(self, x: torch.Tensor, d: int | None = None):
        if d is None:
            d = x.shape[-1]
        mu = self.exponents()
        a = self.a
        r = compute_radius(x)

        r_pow = safe_pow(r, mu)
        f = r_pow @ a

        r_pow_m2 = safe_pow(r, mu - 2)
        grad_coeff = (r_pow_m2 * mu * a).sum(dim=-1)
        grad_f = grad_coeff.unsqueeze(-1) * x

        lap_coeff = mu * (mu + d - 2)
        lap_f = (r_pow_m2 * lap_coeff * a).sum(dim=-1)

        if self.include_log:
            mu_log = self.log_exponent()
            f = f + self.c_log * log_primitive(r, mu_log)
            eps = 1e-12
            r_safe = r + eps
            grad_f = grad_f + self.c_log * (x / (r_safe**2).unsqueeze(-1))
            lap_f = lap_f + self.c_log * (d - 2) / (r_safe**2)

        return f, grad_f, lap_f


class RadialAngularMuntzEdge(nn.Module):
    """Radial + angular modes with learnable radial exponents for both branches."""

    def __init__(self, K_r: int = 6, K_a: int = 4, L_max: int = 2, dim: int = 2,
                 include_log: bool = True, init_scale: float = 0.02, mu_min: float = -2.0, mu_max: float = 4.0):
        super().__init__()
        self.K_r, self.K_a = K_r, K_a
        self.L_max = L_max
        self.dim = dim
        self.mu_min, self.mu_max = mu_min, mu_max
        self.include_log = include_log

        self.n_angular = num_spherical_harmonics(L_max, dim)

        self.a_r = nn.Parameter(init_scale * torch.randn(K_r))
        self.a_ang = nn.Parameter(init_scale * torch.randn(self.n_angular, K_a))

        self.raw_mu_r = nn.Parameter(torch.randn(K_r))
        self.raw_mu_a = nn.Parameter(torch.randn(K_a))

        if include_log:
            self.c_log = nn.Parameter(torch.tensor(0.0))
            self.raw_mu_log = nn.Parameter(torch.tensor(0.0))

    def exponents(self):
        mu_r = make_ordered_bounded_signed(self.raw_mu_r, self.mu_min, self.mu_max)
        mu_a = make_ordered_bounded_signed(self.raw_mu_a, self.mu_min, self.mu_max)
        return mu_r, mu_a

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = compute_radius(x)
        mu_r, mu_a = self.exponents()

        out = safe_pow(r, mu_r) @ self.a_r

        if self.dim == 2:
            Y = spherical_harmonics_2d(x, self.L_max)
        elif self.dim == 3:
            Y = spherical_harmonics_3d(x, self.L_max)
        else:
            raise ValueError(f"dim={self.dim} not supported")

        r_pow_a = safe_pow(r, mu_a)
        radial_coeff = r_pow_a @ self.a_ang.T
        out = out + (radial_coeff * Y).sum(dim=-1)

        if self.include_log:
            mu_log = 0.5 * torch.tanh(self.raw_mu_log)
            out = out + self.c_log * log_primitive(r, mu_log)

        return out


class MultiCenterRadialEdge(nn.Module):
    """Multi-center superposition with optional per-center log-primitive."""

    def __init__(self, J: int = 2, K: int = 6, d: int = 2, learn_centers: bool = True,
                 include_log: bool = True, init_scale: float = 0.02, mu_min: float = -2.0, mu_max: float = 4.0):
        super().__init__()
        self.J, self.K, self.d = J, K, d
        self.mu_min, self.mu_max = mu_min, mu_max
        self.include_log = include_log

        self.a = nn.Parameter(init_scale * torch.randn(J, K))
        centers_init = torch.randn(J, d) * 0.3
        if learn_centers:
            self.centers = nn.Parameter(centers_init)
        else:
            self.register_buffer("centers", centers_init)

        self.raw_mu = nn.Parameter(torch.randn(J, K))

        if include_log:
            self.c_log = nn.Parameter(torch.zeros(J))
            self.raw_mu_log = nn.Parameter(torch.zeros(J))

    def exponents(self) -> torch.Tensor:
        return torch.stack([make_ordered_bounded_signed(self.raw_mu[j], self.mu_min, self.mu_max) for j in range(self.J)], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = self.exponents()
        diff = x.unsqueeze(-2) - self.centers
        r_j = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-12)

        out = torch.zeros(x.shape[:-1], device=x.device, dtype=x.dtype)
        for j in range(self.J):
            out = out + (safe_pow(r_j[..., j], mu[j]) @ self.a[j])
            if self.include_log:
                mu_log_j = 0.5 * torch.tanh(self.raw_mu_log[j])
                out = out + self.c_log[j] * log_primitive(r_j[..., j], mu_log_j)
        return out
