# Wrappers classes for comparison in benchmarks
import torch

from .RISE.explanations import RISE


class RISEWrapper():
    """
    Wrapper for RISE: Wrap RISE method to allow similar usage in scripts
    """
    def __init__(self, model, n_masks=4000, input_size=224, batch_size=2, **kwargs):
        """
        initialisation of the class
        :param model: model used for the maps computations
        :param n_masks: number of masks used
        :param input_size: input size in pixels
        :param batch_size: batch size for the perturbations
        """
        self.rise = RISE(model, (input_size, input_size), gpu_batch=batch_size)
        self.rise.generate_masks(N=n_masks, s=8, p1=0.1)
        self.input_size = input_size

    def __call__(self, x, class_idx=None):
        """
        Call the saliency method
        :param x: input image tensor
        :param class_idx: index of the class to explain
        :return: a saliency map in shape (input_size, input_size)
        """
        with torch.no_grad():
            return self.rise(x)[class_idx].view(self.input_size, self.input_size).detach()
