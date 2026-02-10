import torch
import torch.nn.functional as F

def make_ordered_cumulative(raw, p_min=0.0, p_max=4.0, min_gap=0.01):
    """Ordered, bounded exponents via differentiable cumulative gaps."""
    gaps = F.softplus(raw) + min_gap
    cumsum = torch.cumsum(gaps, dim=-1)
    max_val = cumsum[..., -1:]
    normalized = cumsum / (max_val + 1e-8)
    return p_min + (p_max - p_min) * normalized

def make_ordered_bounded_signed(raw, p_min=-2.0, p_max=4.0, min_gap=0.05):
    """Ordered, bounded signed exponents in [p_min, p_max] via cumulative gaps."""
    gaps = F.softplus(raw) + min_gap
    cumsum = torch.cumsum(gaps, dim=-1)
    max_val = cumsum[..., -1:]
    normalized = cumsum / (max_val + 1e-8)
    return p_min + (p_max - p_min) * normalized

# Legacy (sorting-based) variants kept for compatibility with older experiments.
def make_bounded_signed(raw, p_min=-2.0, p_max=4.0, eps=1e-3):
    u = torch.sigmoid(raw)
    u_sorted, _ = torch.sort(u)
    return p_min + eps + (p_max - p_min - 2*eps) * u_sorted

def make_ordered_bounded(raw, p_max=4.0, eps=1e-3):
    u = torch.sigmoid(raw)
    u_sorted, _ = torch.sort(u)
    return eps + (p_max - 2*eps) * u_sorted
