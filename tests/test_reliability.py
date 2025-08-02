import torch
from torch.distributions import Normal, kl_divergence

from model.encoders import ReliabilityHead


def test_reliability_detects_noise():
    head = ReliabilityHead(1)
    with torch.no_grad():
        head.proj.weight.fill_(-1.0)
        head.proj.bias.zero_()
    clean = torch.zeros(32, 1)
    noisy = torch.ones(32, 1) * 5.0
    r_clean, _, _ = head(clean)
    r_noisy, _, _ = head(noisy)
    assert r_noisy.mean() < r_clean.mean()


def test_kl_reliability():
    mu1 = torch.tensor([1.0])
    log_sigma1 = torch.tensor([0.0])
    q1 = Normal(mu1, torch.exp(log_sigma1))
    p = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    kl1 = kl_divergence(q1, p)
    q0 = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
    kl0 = kl_divergence(q0, p)
    assert kl1 > 0
    assert kl0 < kl1
