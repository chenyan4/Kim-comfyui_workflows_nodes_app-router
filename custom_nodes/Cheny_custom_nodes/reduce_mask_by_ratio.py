import torch
import numpy as np
from PIL import Image

def tensor2np(image):
    if len(image.shape)<3:
        image=image.unsqueeze(0)
    image_np=(image[0].cpu().numpy()*255).astype(np.uint8)
    return image_np

def tensor2pil(image):
    if len(image.shape)<3:
        image=image.unsqueeze(0)
    image_pil=Image.fromarray((image[0].cpu().numpy()*255).astype(np.uint8))
    return image_pil

def mask2tensor(image):
    new_mask=image.convert("L")
    new_np=np.array(new_mask).astype(np.float32)/255.0
    new_tensor=torch.tensor(new_np)
    return new_tensor.unsqueeze(0)

def np2tensor(image):
    image_tensor=torch.tensor(image.astype(np.float32)/255.0)
    return image_tensor.unsqueeze(0)

class ReduceMaskByRatio:
    CATEGORY="My Nodes/Reduce Mask By Ratio"
    RETURN_TYPES=("MASK","BOX","INT","INT","INT","INT",)
    RETURN_NAMES=("mask","crop_box","x","y","width","height",)
    FUNCTION="reduce_mask_by_ratio"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required":{"mask":("MASK",),},
            "optional":{
                "method": (["Original","Center Width/Height","Face Width/Height"], {"default": "Original"}),
                "ratio": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "x_ratio": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "y_ratio": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
                "up_y_ratio": ("FLOAT", {
                    "default": 0.1,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "number"
                }),
            }
        }

    def reduce_mask_by_ratio(self,mask,method,ratio,x_ratio,y_ratio,up_y_ratio):
        mask_np=tensor2np(mask)
        mask_pil=tensor2pil(mask).convert("L")
        mask_bbox=mask_pil.getbbox()

        if method=="Original":
            # PIL getbbox 返回 (left, upper, right, lower)
            x1,y1,x2,h2 = mask_bbox
            w,h = x2-x1, h2-y1
            reduce_w,reduce_h = w*ratio,h*ratio
            # 在原有 bbox 中心缩小
            new_w,new_h=int(w-reduce_w),int(h-reduce_h)
            new_x,new_y=int(x1 + reduce_w/2),int(y1 + reduce_h/2)
            new_mask=Image.new("L",size=mask_pil.size,color=0)
            paste_mask=Image.new("L",size=(new_w,new_h),color=255)

            new_mask.paste(paste_mask,(new_x,new_y))

            new_mask=mask2tensor(new_mask)
            return (new_mask,(new_x,new_y,new_w,new_h),new_x,new_y,new_w,new_h,)
        
        elif method=='Center Width/Height':
            x1,y1,x2,h2 = mask_bbox
            w,h = x2-x1, h2-y1
            reduce_w,reduce_h=w*x_ratio,h*y_ratio
            new_w,new_h=int(w-reduce_w),int(h-reduce_h)
            new_x,new_y=int(x1 + reduce_w/2),int(y1 + reduce_h/2)
            new_mask=Image.new("L",size=mask_pil.size,color=0)
            paste_mask=Image.new("L",size=(new_w,new_h),color=255)

            new_mask.paste(paste_mask,(new_x,new_y))

            new_mask=mask2tensor(new_mask)
            return (new_mask,(new_x,new_y,new_w,new_h),new_x,new_y,new_w,new_h,)
        
        elif method=='Face Width/Height':
            x1,y1,x2,h2 = mask_bbox
            w,h = x2-x1, h2-y1
            reduce_w,reduce_h=w*x_ratio,h*up_y_ratio
            new_w,new_h=int(w-reduce_w),int(h-reduce_h)
            # Face模式：从顶部开始，只减少上方和两侧
            new_x,new_y=int(x1 + reduce_w/2),int(y1 + reduce_h)
            new_mask=Image.new("L",size=mask_pil.size,color=0)
            paste_mask=Image.new("L",size=(new_w,new_h),color=255)

            new_mask.paste(paste_mask,(new_x,new_y))

            new_mask=mask2tensor(new_mask)
            return (new_mask,(new_x,new_y,new_w,new_h),new_x,new_y,new_w,new_h,)

NODE_CLASS_MAPPINGS={
    "ReduceMaskByRatio":ReduceMaskByRatio
}
NODE_DISPLAY_NAME_MAPPINGS={
    "ReduceMaskByRatio":"Reduce Mask By Ratio(My Nodes)"
}

    