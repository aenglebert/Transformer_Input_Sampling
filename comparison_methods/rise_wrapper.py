# Wrappers classes for comparison in benchmarks
import torch

from .RISE.explanations import RISE


class RISEWrapper():
    def __init__(self, model, n_masks=4000, input_size=224, batch_size=2, **kwargs):
        self.rise = RISE(model, (input_size, input_size), gpu_batch=batch_size)
        self.rise.generate_masks(N=n_masks, s=8, p1=0.1)
        self.input_size = input_size

    def __call__(self, x, class_idx=None):
        with torch.no_grad():
            return self.rise(x)[class_idx].view(self.input_size, self.input_size).detach()
