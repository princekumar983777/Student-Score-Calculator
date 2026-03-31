from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch import nn


class FullyConnectedRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass(frozen=True)
class AnnTrainResult:
    model: FullyConnectedRegressor
    best_val_loss: float
    epochs_trained: int


def resolve_device(device: str) -> torch.device:
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(x.astype(np.float32, copy=False)).to(device)


def train_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    hidden_layers: List[int],
    dropout: float,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> AnnTrainResult:
    dev = resolve_device(device)
    model = FullyConnectedRegressor(input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout=dropout).to(dev)
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))

    Xtr = _to_tensor(X_train, dev)
    ytr = _to_tensor(y_train.reshape(-1), dev)
    Xva = _to_tensor(X_val, dev)
    yva = _to_tensor(y_val.reshape(-1), dev)

    n = Xtr.shape[0]
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=dev)
        epoch_losses: List[float] = []
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb = Xtr[idx]
            yb = ytr[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_loss = float(loss_fn(val_pred, yva).detach().cpu().item())

        if val_loss < best_val - 1e-10:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if patience and bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return AnnTrainResult(model=model, best_val_loss=best_val, epochs_trained=epoch)


@torch.no_grad()
def predict_ann(model: FullyConnectedRegressor, X: np.ndarray, device: str) -> np.ndarray:
    dev = resolve_device(device)
    model = model.to(dev)
    model.eval()
    Xt = _to_tensor(X, dev)
    pred = model(Xt).detach().cpu().numpy()
    return pred

