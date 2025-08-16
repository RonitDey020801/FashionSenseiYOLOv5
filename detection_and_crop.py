import torch
from PIL import Image

model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s_deepfashion.pt')

results = model('wardrobe_image.jpeg')
results.save()  # saves image with bounding boxes

# Get crops
crops = results.crop(save=False)  # list of numpy arrays