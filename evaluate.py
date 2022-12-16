import torch

import numpy as np

import hydra
from hydra.utils import instantiate

from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

import random

from torchvision.transforms import Resize

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


# Use Hydra to allow easy configuration swap for comparison of methods
@hydra.main(version_base="1.3", config_path="config", config_name="evaluate")
def main(cfg: DictConfig):

    seed_everything(cfg.seed)

    # Get model
    print("Loading model:", cfg.model.name, end="\n\n")
    model = instantiate(cfg.model.init).cuda()
    model.eval()

    if not cfg.metric.npz_only:
        # Get method
        print("Initializing saliency method:", cfg.method.name, end="\n\n")
        method = instantiate(cfg.method.init, model)

    # Get saliencies from npz
    print("Loading saliency maps from", cfg.input_npz, end="\n\n")
    saliency_maps = torch.tensor(np.load(cfg.input_npz)['arr_0'])

    # Get dataset
    print("Loading dataset", end="\n\n")
    dataset = instantiate(cfg.dataset)

    # Get metric
    metric = instantiate(cfg.metric.init, model)

    # Set resize transformation for the saliency maps if upsampling is required
    upsampling_fn = Resize(dataset[0][0].shape[-2:])

    assert len(dataset) == len(saliency_maps), "The saliency maps and the dataset don't have the same number of items"

    # Loop over the dataset to generate the saliency maps
    for (image, class_idx), saliency_map in tqdm(zip(dataset, saliency_maps),
                                                 desc="Computing saliency maps",
                                                 total=len(dataset)):
        image = image.unsqueeze(0).cuda()
        saliency_map = saliency_map.reshape((1, 1, *saliency_map.shape))

        if saliency_map.shape != image.shape:
            saliency_map = upsampling_fn(saliency_map)

        metric(image, saliency_map, class_idx=class_idx)


if __name__ == "__main__":
    main()
