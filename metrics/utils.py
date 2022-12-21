import torch
import torchvision

import math


def blur_baseline(input_image):
    """
    Blur baseline for perturbation metrics
    :param input_image: image used in the metric
    :return: blurred version of the image
    """
    baseline = torchvision.transforms.functional.gaussian_blur(input_image, kernel_size=[11, 11], sigma=[5,5])
    return baseline


def random_baseline(input_image):
    """
    Random baseline for perturbation metrics
    :param input_image: image used in the metric
    :return: random map with the shape of the image
    """
    baseline = torch.rand_like(input_image)
    return baseline


def mean_baseline(input_image):
    """
    Mean baseline for perturbation metrics
    :param input_image: image used in the metric
    :return: mean map with the shape of the image
    """
    baseline = torch.ones_like(input_image) * input_image.mean()
    return baseline


def black_baseline(input_image):
    """
    Black baseline for perturbation metrics
    :param input_image: image used in the metric
    :return: black map with the shape of the image
    """
    baseline = torch.zeros_like(input_image)
    return baseline


# Dictionary of baseline generation functions
baseline_fn_dict = {
    "blur": blur_baseline,
    "random": random_baseline,
    "mean": mean_baseline,
    "black": black_baseline,
}


class RelevanceMetric:
    """
    Base class for insertion and deletion metrics
    """
    def __init__(self, model, n_steps, batch_size, baseline="blur", softmax=True):
        self.model = model
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.baseline_fn = baseline_fn_dict[baseline]
        self.softmax = softmax

    def __call__(self, image, saliency_map, target=None):
        """
        Metric computation
        :param image: Image for which to compute the metric
        :param saliency_map: Saliency map related to the image
        :param target: Class of the saliency metric
        :return: detailed scores for every steps
        """
        assert image.shape[-2:] == saliency_map.shape[-2:], "Image and saliency map should have the same resolution"

        assert isinstance(target, int), "Target class should be specified for the metric"

        # Get image resolution
        h, w = image.shape[-2:]

        # Generate baseline
        baseline = self.baseline_fn(image)

        # Index of pixels in the saliency map in descending order
        sorted_index = torch.flip(saliency_map.view(-1, h * w).argsort(), dims=[-1])

        # Number of pixels to add or remove per step
        pixels_per_steps = math.ceil((h * w) / self.n_steps)

        # Generate masked samples
        samples = self.generate_samples(sorted_index, image, baseline)

        # Scores tensor
        scores = torch.zeros(self.n_steps).to(image.device)

        # Loop over the dataset by batches
        for idx in range(math.ceil(samples.shape[0] / self.batch_size)):
            # Selection slice for the current batch
            selection_slice = slice(idx * self.batch_size, min((idx + 1) * self.batch_size, samples.shape[0]))

            # Forward pass for the current batch of masked images
            res = self.model(samples[selection_slice])
            # Apply sofmax if required
            if self.softmax:
                res = torch.softmax(res, dim=1)

            # Add the result for class_idx for the current batch
            scores[selection_slice] = res[:, target].detach()

        # Return auc with detailed scores
        return scores

    def generate_samples(self, *args, **kwargs):
        """
        Method to be implemented by child class
        """
        raise NotImplementedError
