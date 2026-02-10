import torch

def log_r(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
    return torch.log(r)

def r_power(alpha: float):
    def fn(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        r = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
        return r**alpha
    return fn

def coulomb_3d(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
    return 1.0 / r

def dipole_3d(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r = torch.sqrt(torch.sum(x**2, dim=-1) + eps)
    z = x[..., 2]
    return z / (r**3 + eps)

def smooth_2d(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(torch.pi * x[..., 0]) * torch.sin(torch.pi * x[..., 1])
