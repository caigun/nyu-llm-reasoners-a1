import torch.nn as nn
import torch

from collections.abc import Callable, Iterable
from typing import Optional
import math
import os
import typing

from einops import rearrange

import numpy as np

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
        logits -= logits.max(dim=-1, keepdim=True).values
        exp_logits = torch.exp(logits)
        target_logit = logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        loss = torch.log(torch.sum(exp_logits, dim=-1)) - target_logit
        return loss.mean()


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p] # Get state associated with p.
            t = state.get("t", 0) # Get iteration number from the state, or initial value.
            grad = p.grad.data # Get the gradient of loss with respect to p.
            p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
            state["t"] = t + 1 # Increment iteration number.
        return loss
    
class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        assert 0.0 <= betas[0] < 1.0, "Invalid beta1 parameter: {}".format(betas[0])
        assert 0.0 <= betas[1] < 1.0, "Invalid beta2 parameter: {}".format(betas[1])
        assert eps > 0.0, "Invalid epsilon value: {}".format(eps)
        assert weight_decay >= 0.0, "Invalid weight_decay value: {}".format(weight_decay)
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                lr_t = lr * math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
        return loss
    
def cosine_lr_scheduler(t: int, lr_max: float, lr_min: float, T_warmup: int, T_iter: int) -> float:
    if t < T_warmup:
        lr = lr_max * t / T_warmup
        return lr
    elif t >= T_iter:
        return lr_min
    else:
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (t - T_warmup) / (T_iter - T_warmup)))
    return lr

@torch.no_grad()
def gradient_clipping(params, max_norm, eps=1e-6):
    total_norm_sq = 0
    for p in params:
        if p.grad is not None:
            total_norm_sq += p.grad.norm() ** 2

    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(scale)


def load_data(x: np.array, batch_size: int, context_len: int, device: torch.device) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    """
    x: np.array of shape (num_tokens,)
    """
    num_tokens = x.shape[0]
    max_start_index = num_tokens - context_len - 1

    start_indices = np.random.randint(0, max_start_index + 1, size=(batch_size,))
    x_batch = np.zeros((batch_size, context_len), dtype=np.int64)
    y_batch = np.zeros((batch_size, context_len), dtype=np.int64)

    for i, start_idx in enumerate(start_indices):
        x_batch[i] = x[start_idx : start_idx + context_len]
        y_batch[i] = x[start_idx + 1 : start_idx + context_len + 1]

    x_tensor = torch.tensor(x_batch, device=device)
    y_tensor = torch.tensor(y_batch, device=device)

    return x_tensor, y_tensor

def save_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    inp: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> int:
    checkpoint = torch.load(inp)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration