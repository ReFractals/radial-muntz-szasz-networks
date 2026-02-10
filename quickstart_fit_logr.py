"""Quickstart: fit log(r) in 2D with RMNDirect."""
import torch
from rmn.models import RMNDirect
from rmn.training import TrainConfig, fit_function
from rmn.benchmarks import log_r

def main():
    model = RMNDirect(d_in=2, K=12, include_log=True)
    cfg = TrainConfig(steps=2000, lr=2e-3, seed=42, device="cuda" if torch.cuda.is_available() else "cpu")
    res = fit_function(model, target_fn=log_r, d=2, config=cfg)
    print("RMSE:", res["rmse"])

if __name__ == "__main__":
    main()
