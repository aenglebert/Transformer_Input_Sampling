from hydra.utils import instantiate


class CaptumWrapper:
    """
    Wrapper for Captum methods
    """
    def __init__(self, model, method_cfg_list, batch_size=None, method_attrs={}):
        self.model = model
        method = model
        for method_cfg in method_cfg_list:
            method = instantiate(method_cfg, method)
        self.method = method
        self.batch_size = batch_size
        self.method_attrs = method_attrs

    def __call__(self, x, class_idx=None):
        return self.method.attribute(x,
                                     target=class_idx,
                                     **self.method_attrs).sum((0, 1))
