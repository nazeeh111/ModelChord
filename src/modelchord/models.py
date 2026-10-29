"""Small explicit architecture registry. Unknown architectures are rejected."""
from __future__ import annotations
from typing import Any
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], classes: list[str]):
        super().__init__()
        self.spec = {"kind": "mlp", "input_dim": input_dim, "hidden": hidden, "classes": classes}
        dims = [input_dim, *hidden, len(classes)]
        self.layers = nn.ModuleList(nn.Linear(a,b) for a,b in zip(dims[:-1],dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers)-1:
                x = torch.relu(x)
        return x

    def activations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {}
        for i, layer in enumerate(self.layers):
            out[f"layers.{i}.weight"] = x.detach()
            x = layer(x)
            if i < len(self.layers)-1:
                x = torch.relu(x)
        return out

def from_spec(spec: dict[str, Any]) -> MLP:
    if set(spec) != {"kind", "input_dim", "hidden", "classes"} or spec["kind"] != "mlp":
        raise ValueError("Unsupported architecture; only the explicit mlp/v1 registry is supported")
    dims = [spec["input_dim"], *spec["hidden"], len(spec["classes"])]
    if not 2 <= len(dims) <= 8 or any(type(d) is not int or not 1 <= d <= 4096 for d in dims):
        raise ValueError("Invalid or oversized architecture")
    if sum((a+1)*b for a,b in zip(dims[:-1],dims[1:])) > 16_000_000:
        raise ValueError("Architecture exceeds the bounded artifact parameter limit")
    if len(set(spec["classes"])) != len(spec["classes"]) or not all(isinstance(c,str) for c in spec["classes"]):
        raise ValueError("Invalid class list")
    return MLP(spec["input_dim"],spec["hidden"],spec["classes"])
