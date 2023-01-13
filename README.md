# Transformer Input Sampling

## Introduction
This repository contain the source code for Transformer Input Sampling method.
The method produce saliency maps for vision transformers using token masking.
The activations of the network being used to produce binary masks for the tokens.

## Requirements
A requirements.txt file is provided to install the necessary libraries to use this method.

## Requirements for comparison
This repository also hold scripts to benchmark in comparison to other methods.

A script is provided to configure the comparison methods repositories ( comparison_methods/configure.sh ).
The requirements specifics to comparison methods can be installed using requirements_comparison.txt
By default, the imagenet path is "inputs/imagenet/", it can be changed in the hydra dataset config.

## Usage

Exemple of usage

``` 
# Load a ViT model
import timm
model = timm.create_model("vit_base_patch16_224", pretrained=True).cuda()
model.eval()

# Set tranforms, normalise to ImageNet train mean and sd 
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ])

# Get image 
from PIL import Image 
image = Image.open(path/to/image.png).convert('RGB') 
input_tensor = transform(image)

# Initialize the saliency class (adapt the batch_size depending on the available memory)
from tis import TIS
saliency_method = tis(model, batch_size=512)

# class_idx can be omited, in this case the maximum predicted class will be used
saliency_map = tis(image, class_idx=class_idx).cpu()
``` 
