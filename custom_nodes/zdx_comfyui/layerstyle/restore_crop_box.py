from .imagefunc import *

NODE_NAME = 'RestoreCropBox'


def unpad_restore(image, mask, pad):
    # 2 restore pad
    croped_image_unpad = image
    croped_mask_unpad = mask
    if pad is not None:
        # Check if pad is new 4-element format or old 2-element format
        if len(pad) == 4:
            pad_left, pad_top, pad_right, pad_bottom = pad
        else:
            pad_left, pad_top = pad[0], pad[1]
            pad_right, pad_bottom = pad_left, pad_top # Fallback for symmetric padding

        # Handle 0 padding correctly (slice(0, -0) results in empty tensor)
        # Use specific padding values for each side
        end_h = -pad_bottom if pad_bottom > 0 else None
        end_w = -pad_right if pad_right > 0 else None
        
        # Start index is simply top/left padding
        start_h = pad_top
        start_w = pad_left
        
        croped_image_unpad = croped_image_unpad[:, start_h:end_h, start_w:end_w, :]
        if croped_mask_unpad is not None:
            croped_mask_unpad = croped_mask_unpad[:, start_h:end_h, start_w:end_w]
    return (croped_image_unpad, croped_mask_unpad, )

class RestoreCropBox:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):

        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "crop_box": ("BOX",),
            },
            "optional": {
                "croped_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = 'zdx/LayerUtility'

    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_box,
                         croped_mask=None
                         ):

        real_w,real_h = crop_box[2]-crop_box[0], crop_box[3]-crop_box[1]
        orig_w,orig_h = croped_image.shape[2], croped_image.shape[1]
        pad_w = (orig_w - real_w) // 2
        pad_h = (orig_h - real_h) // 2
        # unpad crop_box
        croped_image_unpad = croped_image
        croped_mask_unpad = croped_mask
        if pad_w > 0 or pad_h > 0:
        # 2 restore pad
            # Handle 0 padding correctly (slice(0, -0) results in empty tensor)
            end_h = -pad_h if pad_h > 0 else None
            end_w = -pad_w if pad_w > 0 else None
            croped_image_unpad = croped_image[:, pad_h:end_h, pad_w:end_w, :]
            if croped_mask is not None:
                croped_mask_unpad = croped_mask[:, pad_h:end_h, pad_w:end_w]
        croped_image = croped_image_unpad
        if croped_mask is not None:
            croped_mask = croped_mask_unpad
        b_images = []
        l_images = []
        l_masks = []
        ret_images = []
        ret_masks = []
        for b in background_image:
            b_images.append(torch.unsqueeze(b, 0))
        for l in croped_image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', size=m.size, color='white'))
        if croped_mask is not None:
            if croped_mask.dim() == 2:
                croped_mask = torch.unsqueeze(croped_mask, 0)
            l_masks = []
            for m in croped_mask:
                if invert_mask:
                    m = 1 - m
                l_masks.append(tensor2pil(torch.unsqueeze(m, 0)).convert('L'))

        max_batch = max(len(b_images), len(l_images), len(l_masks))
        for i in range(max_batch):
            background_image = b_images[i] if i < len(b_images) else b_images[-1]
            croped_image = l_images[i] if i < len(l_images) else l_images[-1]
            _mask = l_masks[i] if i < len(l_masks) else l_masks[-1]

            _canvas = tensor2pil(background_image).convert('RGB')
            _layer = tensor2pil(croped_image).convert('RGB')

            ret_mask = Image.new('L', size=_canvas.size, color='black')
            _canvas.paste(_layer, box=tuple(crop_box), mask=_mask)
            ret_mask.paste(_mask, box=tuple(crop_box))
            ret_images.append(pil2tensor(_canvas))
            ret_masks.append(image2mask(ret_mask))

        log(f"{NODE_NAME} Processed {len(ret_images)} image(s).", message_type='finish')
        return (torch.cat(ret_images, dim=0), torch.cat(ret_masks, dim=0),)

class RestoreCropBoxPad(RestoreCropBox):
    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "background_image": ("IMAGE", ),
                "croped_image": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),  # 反转mask#
                "info": ("INFO",),
            },
            "optional": {
                "croped_mask": ("MASK",),
                "crop_box": ("BOX",),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", )
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = 'restore_crop_box'
    CATEGORY = 'zdx/LayerUtility'
    @torch.inference_mode()
    def restore_crop_box(self, background_image, croped_image, invert_mask, crop_box=None,
                         croped_mask=None, info={}):
        pad = info.get("pad", None)
        if crop_box is None:
            crop_box = info.get("crop_box", None)
        if crop_box is None:
            raise ValueError("please use FocusCrop to get crop_box and pad info")

        # 2 restore pad
        croped_image_unpad, croped_mask_unpad = unpad_restore(croped_image, croped_mask, pad)

        return super().restore_crop_box(background_image=background_image, croped_image=croped_image_unpad, 
                invert_mask=invert_mask, crop_box=crop_box, croped_mask=croped_mask_unpad)

