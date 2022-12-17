import torch

from .vit_explain.vit_rollout import VITAttentionRollout


class RolloutWrapper():
    """
    Wrapper for Attention Rollout
    """
    def __init__(self, model, discard_ratio=0.9, head_fusion='mean', **kwargs):
        """
        initialisation of the class
        :param model: model used for the maps computations
        :param n_masks: number of masks used
        :param input_size: input size in pixels
        :param batch_size: batch size for the perturbations
        """

        self.method = VITAttentionRollout(model, discard_ratio=discard_ratio, head_fusion=head_fusion)

    def __call__(self, x, class_idx=None):
        """
        Call the saliency method
        :param x: input image tensor
        :param class_idx: index of the class to explain
        :return: a saliency map in shape (input_size, input_size)
        """
        saliency_map = torch.tensor(self.method(x))
        print(saliency_map)
        return saliency_map
