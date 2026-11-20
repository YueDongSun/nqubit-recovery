"""Recoverable versus irreducible noise in multiplexed W-state memories."""

import torch

# TF32 breaks U(1) gauge invariance (phase error about 2e-3 versus 1e-6 in FP32).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

__version__ = "0.1.0"
