# Transformer Input Sampling (TiS)

## Introduction
This repository contains the source code for Transformer Input Sampling (TiS) method.
The method produces saliency maps for vision transformers using token masking.
The activations of the network are used to produce binary masks for the tokens.

## Requirements
A requirements.txt file is provided to install the necessary libraries to use this method.

## Requirements for comparison
This repository also hold scripts to benchmark in comparison to other methods.

A script is provided to configure the comparison methods repositories ("comparison_methods/configure_comparison.sh").
The requirements specific to comparison methods can be installed using requirements_comparison.txt
By default, the imagenet path is "inputs/imagenet/", it can be changed in the hydra dataset config.

## Usage
The method is provided ready to use as a script, a notebook, or can be used in any project as a library.
You need to install the dependencies listed in Section 'Requirements'.

### Demonstration script
It can be used on an arbitrary image with the following command line:

```python tis_example.py input_file=image.jpg```, by replacing 'image.jpg' with your image.

By default, the result of the image and on overlay of the map is displayed.
Instead of displaying the result, you can save it in a file by using and 'output_file' argument as so:

```python tis_example.py input_file=image.jpg output_file=output.png```

If not specified with a class_idx argument, the class used is the maximum output of the model.

This script uses hydra, so any parameter from the configuration files (in config folder) can be changed in the command line.
Here is an example with a batch size of 16, using a DeiT model and generating an explanation for German shepherd (235).

```python tis_example.py input_file=image.jpg method.init.batch_size=32 model=deit class_idx=235```

Additionally, this script is compatible with the compared methods (see the 'Requirements for comparison' Section), 
for example using RISE:

```python tis_example.py input_file=image.jpg method=rise```

### Notebook
A jupyter notebook is provided as ```TIS_test.ipynb``` and offers the opportunity to play in live with the method.
It requires the Imagenet validation set by default, but can be easily adapted to an arbitrary image.


### Import in any project
The method can be used as a library by importing this repository for your project.
Here is an example displaying a typical usage.

``` 
from torchvision import transforms

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
image = Image.open("dog.png").convert('RGB') 
input_tensor = transform(image).cuda()

# Initialize the saliency class (adapt the batch_size depending on the available memory)
from tis import TIS
saliency_method = TIS(model, batch_size=512)

# class_idx can be omited, in this case the maximum predicted class will be used
saliency_map = saliency_method(input_tensor, 
                   #class_idx=class_idx
                  ).cpu()
``` 
