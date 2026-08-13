from torchvision import models, transforms, datasets
import torch
import urllib.request
from PIL import Image
import torch.nn.functional as F

model = models.resnet18(weights='IMAGENET1K_V1')
model.eval()

image = Image.open()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
categories = models.ResNet18_Weights.IMAGENET1K_V1.meta['categories']

image = transform(image)
input_tensor = torch.unsqueeze(image, 0)
print(input_tensor.shape)

logits = model(input_tensor)[0]

probs = F.softmax(logits, 0)
print(logits[100:110])
print(probs[100:110])

sort = torch.argsort(logits, 0, descending=True)
for i in range(10):
    print(i)
    print(categories[sort[i]])
    print(probs[sort[i]])

print(categories[logits.argmax()])
print(logits.shape)