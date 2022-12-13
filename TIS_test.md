---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: TIS
    language: python
    name: tis
---

```python
import torch
from torch.nn.functional import interpolate

from torchvision import transforms
from torchvision.models import vit_b_16
from torchvision.datasets import ImageNet

import timm

from matplotlib import pyplot as plt

from tis import TIS
```

```python
imagenet_path = "./datasets/imagenet/"
```

```python
def overlay(image, saliency, alpha=0.7):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    image = image.permute(1, 2, 0)
    saliency = interpolate(saliency.reshape((1, 1, *saliency.shape)), size=image.shape[:2], mode='bilinear')
    saliency = saliency.squeeze()
    ax[0].imshow(image)
    ax[1].imshow(image)
    ax[1].imshow(saliency, alpha=alpha, cmap='jet')
    plt.show()
```

```python
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                ])

transform_display = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((256, 256)),
                                transforms.CenterCrop(224),
                                ])
```

```python
dataset = ImageNet(imagenet_path, transform=transform, split="val")
dataset_display = ImageNet(imagenet_path, transform=transform_display, split="val")
```

```python
#vit_model = vit_b_16(weights='IMAGENET1K_V1').cuda()
vit_model = timm.create_model("vit_base_patch16_224", pretrained=True, pretrained_cfg="vit_base_patch16_224.orig_in21k_ft_in1k").cuda()
#vit_model = timm.create_model("deit_base_patch16_224", pretrained=True).cuda()
```

```python
tis = TIS(vit_model, n_masks=2048, batch_size=512)
```

```python
i = int(torch.rand(1).item()*50000)
class_idx = dataset[i][1]

print("idx", i)
print(dataset.classes[class_idx])
image = dataset[i][0].unsqueeze(0).cuda()
saliency = tis(image, class_idx=class_idx).cpu()
overlay(dataset_display[i][0], saliency.cpu(), alpha=0.7)
```

```python
import random

subset_size = 5000

indices = list(range(50000))
random.shuffle(indices)
```

```python
for indice in indices[:subset_size]:
    print("-", indice)
```

```python

```
