import torch
import numpy as np
from PIL import Image

def tensor2np(image):
    if len(image.shape)<3:
        image=image.unsqueeze(0)
    
    return (image.cpu().numpy()*255).astype(np.uint8)

def np2tensor(image):
    image=image.astype(np.float32)/255
    image=torch.tensor(image)
    return image.unsqueeze(0)

class FindCropBox:
    CATEGORY="My Nodes/Find Crop Box"
    RETURN_TYPES=("INT","INT","INT","INT")
    RETURN_NAMES=("x","y","width","height")
    FUNCTION="find_crop_box"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{
                "destination":("IMAGE",),
                "source":("IMAGE",),
            },
        }

    def find_crop_box(self,destination,source):
        destination_np=tensor2np(destination)
        source_np=tensor2np(source)

        # destination_np / source_np: (B, H, W) or (B, H, W, C)
        d_h, d_w = destination_np.shape[1], destination_np.shape[2]
        s_h, s_w = source_np.shape[1], source_np.shape[2]

        h_bound = d_h - s_h
        w_bound = d_w - s_w

        # 在 destination 中滑动窗口，查找与 source 完全相同的区域
        for i in range(h_bound + 1):
            for j in range(w_bound + 1):
                # 取出 destination 中的一个候选区域，保留全部通道维度（若存在）
                patch = destination_np[0, i:i + s_h, j:j + s_w, ...]
                target = source_np[0, :s_h, :s_w, ...]
                if np.array_equal(patch, target):
                    return (j, i, s_w, s_h)
        
        return (0,0,0,0,)

NODE_CLASS_MAPPINGS={
    "FindCropBox":FindCropBox
}

NODE_DISPLAY_NAME_MAPPINGS={
    "FindCropBox":"Find Crop Box(My Node)"
}

        