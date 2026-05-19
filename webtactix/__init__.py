import random
import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    """Set all RNG seeds for reproducibility. Call once at process startup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
