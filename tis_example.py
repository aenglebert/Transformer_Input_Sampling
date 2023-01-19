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