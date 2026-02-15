import torch
from PIL import Image,ImageDraw
import numpy as np



def mask2pil(mask):
    if len(mask.shape)<3:
        mask=mask.unsqueeze(0)
    return Image.fromarray((mask[0].cpu().numpy()*255).astype(np.uint8))


def mask2tensor(mask):
    mask=mask.convert('L')
    mask_array=np.array(mask).astype(np.float32)/255.0
    mask_tensor=torch.tensor(mask_array)
    return mask_tensor

class MaskSolidArea:

    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "mask": ("MASK",),
                    },
                    'optional':{
                        'radius':('INT',{
                            'default':100,
                            'min':0,
                            'max':1000,
                            'steo':1,
                        })
                    }
                        }

    CATEGORY = "My Nodes/mask_solid_area"

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("MASKS",)

    FUNCTION = "cut"
    def cut(self, mask,radius):
        mask_pil = mask2pil(mask)
        mask_bbox = mask_pil.getbbox()
        need_paste_mask=Image.new('L',mask_pil.size,"black")
        if mask_bbox is None:
            return (mask,)
        # new_mask = torch.zeros_like(mask[0], dtype=torch.float32)
        # # new_mask bbox 区域绘制为全1
        # new_mask[mask_bbox[1]:mask_bbox[3], mask_bbox[0]:mask_bbox[2]] = 1.0
        draw=ImageDraw.Draw(need_paste_mask)
        draw.rounded_rectangle(
            [mask_bbox[0],mask_bbox[1],mask_bbox[2],mask_bbox[3]],
            radius=radius,
            fill=255
        )

        new_mask=mask2tensor(need_paste_mask)
        return (new_mask.unsqueeze(0),)

NODE_CLASS_MAPPINGS={
    'MaskSolidArea':MaskSolidArea
}

NODE_DISPLAY_NAME_MAPPINGS={
    'MaskSolidArea':'MaskSolidArea(My Node)'
}