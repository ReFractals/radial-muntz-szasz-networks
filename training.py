import torch
from dataclasses import dataclass
from typing import Callable, Dict

@dataclass
class TrainConfig:
    lr: float = 2e-3
    steps: int = 5000
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sample_uniform_annulus(n: int, d: int = 2, rmin: float = 0.01, rmax: float = 1.0, device: str = "cpu") -> torch.Tensor:
    pts = []
    while True:
        x = (2*torch.rand((n, d), device=device) - 1.0) * rmax
        r = torch.norm(x, dim=-1)
        mask = (r >= rmin) & (r <= rmax)
        pts.append(x[mask])
        cat = torch.cat(pts, dim=0)
        if cat.shape[0] >= n:
            return cat[:n]

def fit_function(
    model: torch.nn.Module,
    target_fn: Callable[[torch.Tensor], torch.Tensor],
    n_train: int = 8192,
    n_test: int = 2048,
    d: int = 2,
    rmin: float = 0.01,
    rmax: float = 1.0,
    config: TrainConfig = TrainConfig(),
    extra_loss: Callable[[torch.nn.Module], torch.Tensor] | None = None,
) -> Dict:
    set_seed(config.seed)
    device = config.device
    model = model.to(device).train()

    x_train = sample_uniform_annulus(n_train, d=d, rmin=rmin, rmax=rmax, device=device)
    y_train = target_fn(x_train).to(device)

    x_test = sample_uniform_annulus(n_test, d=d, rmin=rmin, rmax=rmax, device=device)
    y_test = target_fn(x_test).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    history = []
    log_every = max(1, config.steps // 50)

    for step in range(config.steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x_train).squeeze(-1)
        loss = torch.mean((pred - y_train)**2)
        if extra_loss is not None:
            loss = loss + extra_loss(model)
        loss.backward()
        if config.grad_clip and config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt.step()
        if (step + 1) % log_every == 0:
            history.append(float(loss.detach().cpu()))

    model.eval()
    with torch.no_grad():
        pred = model(x_test).squeeze(-1)
        rmse = torch.sqrt(torch.mean((pred - y_test)**2)).item()

    return {"rmse": rmse, "loss_history": history}
