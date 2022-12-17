import torch

import sys

sys.path.append("comparison_methods/Transition_Attention_Maps")

from baselines.ViT.ViT_new import VisionTransformer, _conv_filter, _cfg
from baselines.ViT.helpers import load_pretrained
from baselines.ViT.interpret_methods import InterpretTransformer

from timm.models.vision_transformer import default_cfgs as vit_cfgs
from timm.models.deit import default_cfgs as deit_cfgs


"""
Models: The TAM method uses modified models.
We changed the model creation functions to use timm weights as defined in the hydra configuration
instead of fixed weights defined by the authors
"""


def vit_base_patch16_224(pretrained=False, model_name="vit_base_patch16_224", pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    cfg = _cfg(
        url=vit_cfgs[model_name].cfgs[pretrained_cfg].url,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    )
    model.default_cfg = cfg
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_large_patch16_224(pretrained=False, model_name="vit_large_patch16_224", pretrained_cfg=None, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, **kwargs)
    cfg = _cfg(
        url=vit_cfgs[model_name].cfgs[pretrained_cfg].url,
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    )
    model.default_cfg = cfg
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


def deit_base_patch16_224(pretrained=False, model_name="deit_base_patch16_224", **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, **kwargs)
    cfg = _cfg(
        url=deit_cfgs[model_name]['url'],
    )
    model.default_cfg = cfg
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=cfg['url'],
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


# Method computation

class TAMWrapper:
    """
    Wrapper for TAM method: Wrap the method to allow similar usage in scripts
    """
    def __init__(self, model, start_layer=0, steps=20):
        """
        initialisation of the class
        :param model: model used for the maps computations
        :param start_layer: start layer of TAM
        :param steps: steps of TAM
        """
        # Check that model is a patched ViT
        assert isinstance(model, VisionTransformer), "Transformer architecture not recognised"

        self.model = model
        self.method = InterpretTransformer(model)
        self.start_layer = start_layer
        self.steps = steps

    def __call__(self, x, class_idx=None):
        """
        Call the saliency method
        :param x: input image tensor
        :param class_idx: index of the class to explain
        :return: a saliency map in shape (input_size, input_size)
        """
        saliency_map = self.method.transition_attention_maps(x,
                                                             index=class_idx,
                                                             start_layer=self.start_layer,
                                                             steps=self.steps,
                                                             )
        return saliency_map.reshape(14, 14)
