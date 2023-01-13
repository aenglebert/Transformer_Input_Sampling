from captum.metrics import sensitivity_max

from torchvision.transforms import Resize


def captum_method_wrapper(method):
    """
    A wrapper for the explainibility method to return saliency in the format expected by captum
    :param method:
    :return:
    """
    def fn(inputs, target=None):
        # Set resize transformation for the saliency maps if upsampling is required
        inputs = inputs[0]
        upsampling_fn = Resize(inputs.shape[-2:])

        # Generate saliency map
        saliency = method(inputs, class_idx=target)

        # Expend saliency map to captum format
        saliency = saliency.view(1, 1, *saliency.shape)

        return saliency

    return fn


class SensitivityMax:
    def __init__(self, model, method, n_perturb_samples=10, perturb_radius=0.02):
        self.model = model
        self.attribute_fn = captum_method_wrapper(method)
        self.n_perturb_samples = n_perturb_samples
        self.perturb_radius = perturb_radius

    def __call__(self, image, target):
        sens = sensitivity_max(self.attribute_fn,
                               image,
                               target=target,
                               max_examples_per_batch=1,
                               n_perturb_samples=self.n_perturb_samples,
                               perturb_radius=self.perturb_radius,
                               )
        return sens
