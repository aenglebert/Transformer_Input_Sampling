import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch
from torch.nn.functional import interpolate
import numpy as np
import random

from datasets.ilsvrc2012 import classes

from PIL import Image

from matplotlib import pyplot as plt


# Define a function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def overlay(image, saliency, alpha=0.7):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    image = image.permute(1, 2, 0)
    saliency = interpolate(saliency.reshape((1, 1, *saliency.shape)), size=image.shape[:2], mode='bilinear')
    saliency = saliency.squeeze()
    ax[0].imshow(image)
    ax[1].imshow(image)
    ax[1].imshow(saliency, alpha=alpha, cmap='jet')
    plt.show()


@hydra.main(version_base="1.3", config_path="config", config_name="example")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Get model
    print("Loading model:", cfg.model.name, end="\n\n")
    model = instantiate(cfg.model.init).to(device)
    model.eval()

    # Get method
    print("Initializing saliency method:", cfg.method.name, end="\n\n")
    method = instantiate(cfg.method.init, model)

    # Get transformations
    print("Setting transformations", end="\n\n")
    transform = instantiate(cfg.transform)

    # Get image
    print("Opening image:", cfg.input_file, end="\n\n")
    image_raw = Image.open(cfg.input_file).convert('RGB')
    image = transform(image_raw).to(device).unsqueeze(0)

    if not cfg.class_idx:
        class_idx = torch.argmax(model(image), dim=-1)[-1]
    else:
        class_idx = cfg.class_idx

    # Computing saliency map
    print("Computing saliency map using", cfg.method.name, "for class", classes[class_idx])
    saliency_map = method(image, class_idx=class_idx).detach().cpu()

    image = image - image.min()
    image = image/image.max()
    overlay(image.cpu(), saliency_map)


if __name__ == "__main__":
    main()
