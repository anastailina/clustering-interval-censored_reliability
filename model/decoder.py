import torch
import torch.nn.functional as F


def scale_noise_by_reliability(r: torch.Tensor, base_sigma2: float = 1.0, g: str = "softplus") -> torch.Tensor:
    """Scale a base variance according to reliability r.

    Parameters
    ----------
    r : Tensor
        Reliability values in (0, 1) with shape ``(batch, 1)``.
    base_sigma2 : float, optional
        Base observation variance. Defaults to ``1.0``.
    g : {"softplus", "exp"}
        Monotone function used to scale the variance. Lower reliability
        results in a larger variance.
    """
    if g == "exp":
        scale = torch.exp(-r)
    else:
        scale = 1.0 + F.softplus(-r)
    return base_sigma2 * scale
