import torch
import torch.nn as nn

class ReliabilityHead(nn.Module):
    """Simple head that parameterises q(r|h).

    The module outputs a mean and log standard deviation for a Normal
    distribution in logit space. A reparameterised sample is pushed through
    a sigmoid so that the final reliability r lies in (0, 1).
    """

    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, 2)

    def forward(self, h: torch.Tensor):
        mu, log_sigma = self.proj(h).chunk(2, dim=-1)
        eps = torch.randn_like(mu)
        logit_r = mu + torch.exp(log_sigma) * eps
        r = torch.sigmoid(logit_r)
        return r, mu, log_sigma
