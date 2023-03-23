import torch

import numpy as np

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import random

import os

# Try to import lovely_tensors
try:
    import lovely_tensors as lt
    lt.monkey_patch()
except ModuleNotFoundError:
    # But not mandatory, pass if lovely tensor is not available
    pass


# Define a function to seed everything
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def create_directory_if_not_exists(filepath):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Use Hydra to allow easy configuration swap for comparison of methods
@hydra.main(version_base="1.3", config_path="config", config_name="generate")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    # Get model
    print("Loading model:", cfg.model.name, end="\n\n")
    model = instantiate(cfg.model.init).cuda()
    model.eval()

    # Get method
    print("Initializing saliency method:", cfg.method.name, end="\n\n")
    method = instantiate(cfg.method.init, model)

    # Get dataset
    print("Loading dataset", end="\n\n")
    dataset = instantiate(cfg.dataset)

    # Keep saliency maps in a list
    saliency_maps_list = []

    # Loop over the dataset to generate the saliency maps
    for image, class_idx in tqdm(dataset, desc="Computing saliency maps"):
        image = image.unsqueeze(0).cuda()

        if cfg.no_target:
            class_idx = None

        # Compute current saliency map
        cur_map = method(image, class_idx=class_idx).detach().cpu()

        # Add the current map to the list of saliency maps
        saliency_maps_list.append(cur_map)

    # Stack into a single tensor
    saliency_maps = torch.stack(saliency_maps_list)

    # Save as a npz
    output_npz = cfg.output_npz
    if cfg.no_target:
        output_npz += ".notarget"
    print("\nSaving saliency maps to file:", cfg.output_npz)
    create_directory_if_not_exists(output_npz)
    np.savez(cfg.output_npz, saliency_maps.cpu().numpy())


if __name__ == "__main__":
    main()
