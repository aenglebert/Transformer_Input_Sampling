import torch

import numpy as np

import copy

import sys

sys.path.append("comparison_methods/chefer1")

from baselines.ViT.ViT_new import VisionTransformer, _conv_filter, _cfg
from baselines.ViT.helpers import load_pretrained

from timm.models.vision_transformer import default_cfgs as vit_cfgs
from timm.models.deit import default_cfgs as deit_cfgs



"""
Models: The Chefer method uses modified models.
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


"""
Method computation

The functions for Chefer2 method applied to ViT are defined in a notebook at
https://github.com/hila-chefer/Transformer-MM-Explainability/blob/main/Transformer_MM_explainability_ViT.ipynb

We have copied them here for lack of being able to import them
"""


# rule 5 from paper
def avg_heads(cam, grad):
    cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
    grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
    cam = grad * cam
    cam = cam.clamp(min=0).mean(dim=0)
    return cam


# rule 6 from paper
def apply_self_attention_rules(R_ss, cam_ss):
    R_ss_addition = torch.matmul(cam_ss, R_ss)
    return R_ss_addition


def generate_relevance(model, input, index=None):
    output = model(input, register_hook=True)
    if index == None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, index] = 1
    one_hot_vector = one_hot
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * output)
    model.zero_grad()
    one_hot.backward(retain_graph=True)

    num_tokens = model.blocks[0].attn.get_attention_map().shape[-1]
    R = torch.eye(num_tokens, num_tokens).cuda()

    for blk in model.blocks:
        grad = blk.attn.get_attn_gradients()
        cam = blk.attn.get_attention_map()
        cam = avg_heads(cam, grad)
        R += apply_self_attention_rules(R.cuda(), cam.cuda()).detach()

    return R[0, 1:]


class Chefer2Wrapper():
    """
    Wrapper for Chefer2 method: Wrap the method to allow similar usage in scripts
    """
    def __init__(self, model):
        """
        initialisation of the class
        :param model: model used for the maps computations
        """
        # Check that model is a patched ViT
        assert isinstance(model, VisionTransformer), "Transformer architecture not recognised"

        self.model = model

    def __call__(self, x, class_idx=None):
        """
        Call the saliency method
        :param x: input image tensor
        :param class_idx: index of the class to explain
        :return: a saliency map in shape (input_size, input_size)
        """
        with torch.enable_grad():
            saliency_map = generate_relevance(self.model, x, index=class_idx)

            for block in self.model.blocks:
                block.attn.attn_gradients = None
                block.attn.attention_maps = None

            return saliency_map.reshape(14, 14)
