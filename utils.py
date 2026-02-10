import torch

def safe_pow(r: torch.Tensor, mu: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Stable r**mu using exp(mu * log r). r: [...], mu: [K] -> [..., K]."""
    r_safe = torch.clamp(r, min=eps)
    return torch.exp(torch.log(r_safe)[..., None] * mu)

def safe_pow_signed(x: torch.Tensor, mu: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    ax = torch.abs(x) + eps
    return torch.exp(torch.log(ax)[..., None] * mu)

def compute_radius(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.sqrt(torch.sum(x**2, dim=-1) + eps)

def compute_unit_vector(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r = compute_radius(x, eps).unsqueeze(-1)
    return x / r

def log_primitive(r: torch.Tensor, mu: torch.Tensor, eps_mu: float = 1e-4, eps_r: float = 1e-12) -> torch.Tensor:
    r_safe = torch.clamp(r, min=eps_r)
    log_r = torch.log(r_safe)
    if isinstance(mu, (int, float)):
        mu = torch.tensor(mu, device=r.device, dtype=r.dtype)

    abs_mu = torch.abs(mu)
    r_pow = torch.exp(log_r * mu)
    standard = (r_pow - 1) / (mu + 1e-20)

    log_r_sq = log_r ** 2
    log_r_cu = log_r ** 3
    taylor = log_r + mu * log_r_sq / 2 + mu**2 * log_r_cu / 6

    weight = torch.sigmoid((abs_mu - eps_mu) * 100)
    return weight * standard + (1 - weight) * taylor

def spherical_harmonics_2d(x: torch.Tensor, L_max: int = 4) -> torch.Tensor:
    theta = torch.atan2(x[..., 1], x[..., 0] + 1e-12)
    modes = [torch.ones_like(theta)]
    for ell in range(1, L_max + 1):
        modes.append(torch.cos(ell * theta))
        modes.append(torch.sin(ell * theta))
    return torch.stack(modes, dim=-1)

def spherical_harmonics_3d(x: torch.Tensor, L_max: int = 2) -> torch.Tensor:
    x_norm = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-12)
    X, Y, Z = x_norm[..., 0], x_norm[..., 1], x_norm[..., 2]
    modes = [torch.ones_like(X)]
    if L_max >= 1:
        modes.extend([Y, Z, X])
    if L_max >= 2:
        modes.extend([X * Y, Y * Z, 3*Z**2 - 1, X * Z, X**2 - Y**2])
    if L_max >= 3:
        modes.extend([Y * (3*X**2 - Y**2), X * Y * Z, Y * (5*Z**2 - 1), Z * (5*Z**2 - 3),
                      X * (5*Z**2 - 1), Z * (X**2 - Y**2), X * (X**2 - 3*Y**2)])
    return torch.stack(modes, dim=-1)

def num_spherical_harmonics(L_max: int, dim: int = 3) -> int:
    if dim == 2:
        return 2 * L_max + 1
    if dim == 3:
        return (L_max + 1) ** 2
    raise ValueError(f"dim={dim} not supported")
