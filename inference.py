from PIL import Image
import requests
from torchvision import transforms
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

from dinov2.models.vision_transformer import vit_small
from dinov2.models.vision_transformer import DinoVisionTransformer

model = vit_small()
for k,v in model.named_parameters():
    print(k,",its shape is:-",v.shape)

transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

pixel_values = transformations(image).unsqueeze(0)
dino = DinoVisionTransformer()
output = dino.forward_features(x = pixel_values)

for i,j in output.items():
    if isinstance(j, torch.Tensor):
        print(i,j.shape)
        
print(output["x_norm_clstoken"].shape)