from captum.metrics import sensitivity_max

from torchvision.transforms import Resize


def captum_method_wrapper(method):
    """
    A wrapper for the explainibility method to return saliency in the format expected by captum
    :param method: The Captum xai method to wrap
    :return: wrapped Captum method
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
    """
    Sensitivity Max metric class, use Captum function under the hood
    """
    def __init__(self, model, method, n_perturb_samples=10, perturb_radius=0.02):
        """
        Init
        :param model: model to explain
        :param method: xai method to generate saliency maps to evaluate
        :param n_perturb_samples: number of perturbations samples for sensitivity max metric
        :param perturb_radius: radius of the perturbation for sensitivity max
        """
        self.model = model
        self.attribute_fn = captum_method_wrapper(method)
        self.n_perturb_samples = n_perturb_samples
        self.perturb_radius = perturb_radius

    def __call__(self, image, target):
        """
        call the metric
        :param image: input to use
        :param target: target class for the xai method
        :return:
        """
        sens = sensitivity_max(self.attribute_fn,
                               image,
                               target=target,
                               max_examples_per_batch=1,
                               n_perturb_samples=self.n_perturb_samples,
                               perturb_radius=self.perturb_radius,
                               )
        return sens
