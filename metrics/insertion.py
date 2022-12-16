import torch

import math

from .utils import RelevanceMetric


class Insertion(RelevanceMetric):
    def __init__(self, model, n_steps, batch_size, baseline="blur"):
        """
        Init of Insertion metric class
        :param model: model to compute the scores
        :param n_steps: number of steps of the metric
        :param batch_size: batch size for the computations
        :param baseline: replacement baseline when a pixels is masked
        """
        super(Insertion, self).__init__(model, n_steps, batch_size, baseline=baseline)

    def generate_samples(self, indexes, image, baseline):
        """
        Generate the masked images samples
        :param indexes: indexes of pixels in descending order of importance
        :param image: image to mask
        :param baseline: baseline image to replace the masked pixels
        :return: tensor masked images
        """
        h, w = image.shape[-2:]
        pixels_per_steps = math.ceil((h * w) / self.n_steps)
        samples = torch.ones(self.n_steps, *image.shape[-3:]).to(image.device)
        samples = samples * baseline
        for step in range(self.n_steps):
            pixels = indexes[:, :pixels_per_steps*(step+1)]
            samples[step].view(-1, h * w)[..., pixels] = image.view(-1, h * w)[..., pixels]
        return samples
