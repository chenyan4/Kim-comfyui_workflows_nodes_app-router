import torch 
import os

from PIL import Image,ImageDraw
import glob
import numpy as np
import cv2
from ultralytics import YOLO

def tensor2pil(image):
    if len(image.shape)<3:
        image=image.unsqueeze(0)
    return Image.fromarray((image[0].cpu().numpy()*255).astype(np.uint8))

def pil2tensor(image):
    new_image=image.convert('RGB')
    new_array=np.array(new_image).astype(np.float32)/255.-
    new_tensor=torch.tensor(new_array)
    return new_tensor.unsqueeze(0)

