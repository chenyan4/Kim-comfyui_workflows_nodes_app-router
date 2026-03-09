import os
import random
import sys
import gc
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
from PIL import Image
import types
from functools import wraps

# no depedn on comfyui and custom_nodes


def support_pil_image(original_method):
    @wraps(original_method)
    def wrapper(self, image=None, *args, **kwargs):
        if hasattr(image, "save"):
            channel = kwargs.get("channel", None)
            image = image.convert("RGB" if channel is None else "L")
            return (pil2tensor(image),)
        return original_method(self, image=image, *args, **kwargs)
    return wrapper


def pil2tensor(image):
    new_image = image.convert("RGB")
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor


def tensor2pil(image):
    if len(image.shape) < 3:
        image = image.unsqueeze(0)
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    if path is None:
        path = os.getcwd()
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        pass
        return path_name
    parent_directory = os.path.dirname(path)
    if parent_directory == path:
        return None
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = "data/chenyan/comfyui"
    if not os.path.exists(comfyui_path):
        comfyui_path = find_path("comfyui")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        pass


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        pass
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        pass


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    import os
    if os.environ.get("COMFYUI_NODES_LOADED") == "1":
        return
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes

    sys.path.insert(0, find_path("comfyui"))
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    asyncio.run(init_extra_nodes())
    os.environ["COMFYUI_NODES_LOADED"] = "1"


import_custom_nodes()
from nodes import NODE_CLASS_MAPPINGS

# 节点实例（与 qwen_change_bg 风格一致，模块级缓存）
cr_text_node = NODE_CLASS_MAPPINGS["CR Text"]()
getimagesize_node = NODE_CLASS_MAPPINGS["GetImageSize+"]()
impactcompare_node = NODE_CLASS_MAPPINGS["ImpactCompare"]()
layerutility_numbercalculator_node = NODE_CLASS_MAPPINGS["LayerUtility: NumberCalculator"]()
imagescalebyaspectratiov2_node = NODE_CLASS_MAPPINGS["ImageScaleByAspectRatioV2"]()
easy_ifelse_node = NODE_CLASS_MAPPINGS["easy ifElse"]()
rmbg_node = NODE_CLASS_MAPPINGS["RMBG"]()
imagecompositemasked_node = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
imagecrop_node = NODE_CLASS_MAPPINGS["ImageCrop"]()
yolov8_person_nomask_node = NODE_CLASS_MAPPINGS["Yolov8_person_nomask"]()
clothessegment_node = NODE_CLASS_MAPPINGS["ClothesSegment"]()
maskcomposite_node = NODE_CLASS_MAPPINGS["MaskComposite"]()
focuscropultra_node = NODE_CLASS_MAPPINGS["FocusCropUltra"]()
layerutility_cropboxresolve_node = NODE_CLASS_MAPPINGS["LayerUtility: CropBoxResolve"]()
painterfluximageedit_node = NODE_CLASS_MAPPINGS["PainterFluxImageEdit"]()
lanpaint_ksampler_node = NODE_CLASS_MAPPINGS["LanPaint_KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
jwimageresize_node = NODE_CLASS_MAPPINGS["JWImageResize"]()
invertmask_node = NODE_CLASS_MAPPINGS["InvertMask"]()
layerutility_imagescalebyaspectratio_v2_node = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
imagemaskscaleasv2_node = NODE_CLASS_MAPPINGS["ImageMaskScaleAsV2"]()


class flux2_klein_change_bg:
    def __init__(self):
        # 预加载模型，风格对齐 qwen_change_bg / flux2_klein_change_bg
        self.cliploader_88 = NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(
            clip_name="qwen_3_8b_fp8mixed.safetensors",
            type="flux2",
            device="default",
        )
        self.vaeloader_86 = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name="flux2-vae.safetensors")
        self.unetloader_89 = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name="flux/flux-2-klein-9b-fp8.safetensors", weight_dtype="default"
        )
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self, init_image: Image.Image, bg_image: Image.Image) -> Image.Image:
        """
        前景图 init_image + 背景图 bg_image -> FLUX2 流水线换背景后的 PIL.Image
        """
        cr_text_13 = cr_text_node.text_multiline(
            text="Give your subject natural light and shadow effects, blending it seamlessly with its surroundings while maintaining a consistent perspective between the subject and the background."
        )
        cr_text_48 = cr_text_node.text_multiline(
            text="Give your subject natural light and shadow effects, blending it seamlessly with its surroundings while maintaining a consistent perspective between the subject and the background."
        )
        loadimage_78 = self.loadimage.load_image(image=init_image)
        loadimage_79 = self.loadimage.load_image(image=bg_image)

        getimagesize_55 = getimagesize_node.execute(image=get_value_at_index(loadimage_78, 0))
        getimagesize_69 = getimagesize_node.execute(image=get_value_at_index(loadimage_79, 0))

        impactcompare_64 = impactcompare_node.doit(
            cmp="a <= b",
            a=get_value_at_index(getimagesize_69, 0),
            b=get_value_at_index(getimagesize_69, 1),
        )
        layerutility_numbercalculator_60 = (
            layerutility_numbercalculator_node.number_calculator_node(
                operator="max",
                a=get_value_at_index(getimagesize_55, 0),
                b=get_value_at_index(getimagesize_55, 1),
            )
        )
        imagescalebyaspectratiov2_67 = imagescalebyaspectratiov2_node.image_scale_by_aspect_ratio(
            aspect_ratio="original",
            proportional_width=1,
            proportional_height=1,
            fit="letterbox",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=get_value_at_index(layerutility_numbercalculator_60, 0),
            background_color="#000000",
            image=get_value_at_index(loadimage_79, 0),
        )
        imagescalebyaspectratiov2_65 = imagescalebyaspectratiov2_node.image_scale_by_aspect_ratio(
            aspect_ratio="original",
            proportional_width=1,
            proportional_height=1,
            fit="letterbox",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="shortest",
            scale_to_length=get_value_at_index(layerutility_numbercalculator_60, 0),
            background_color="#000000",
            image=get_value_at_index(loadimage_79, 0),
        )
        easy_ifelse_73 = easy_ifelse_node.execute(
            boolean=get_value_at_index(impactcompare_64, 0),
            on_true=get_value_at_index(imagescalebyaspectratiov2_67, 0),
            on_false=get_value_at_index(imagescalebyaspectratiov2_65, 0),
        )
        rmbg_62 = rmbg_node.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Alpha",
            background_color="#222222",
            image=get_value_at_index(loadimage_78, 0),
        )
        imagecompositemasked_57 = imagecompositemasked_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(easy_ifelse_73, 0),
            source=get_value_at_index(rmbg_62, 0),
            mask=get_value_at_index(rmbg_62, 1),
        )
        imagecrop_112 = imagecrop_node.EXECUTE_NORMALIZED(
            width=get_value_at_index(getimagesize_55, 0),
            height=get_value_at_index(getimagesize_55, 1),
            x=0,
            y=0,
            image=get_value_at_index(imagecompositemasked_57, 0),
        )
        yolov8_person_nomask_90 = yolov8_person_nomask_node.yolov8_person_nomask(
            yolo_model="yolo11m-seg.pt",
            true_rate=0.1,
            img_ratio=0.6666666666666666,
            x_ratio=0.5,
            y_ratio=0.1,
            radius=100,
            blur_radius=0,
            back_image=get_value_at_index(imagecrop_112, 0),
        )
        clothessegment_91 = clothessegment_node.segment_clothes(
            Hat=True,
            Hair=False,
            Face=False,
            Sunglasses=True,
            Upper_clothes=False,
            Skirt=False,
            Dress=False,
            Belt=False,
            Pants=False,
            Left_arm=False,
            Right_arm=False,
            Left_leg=False,
            Right_leg=False,
            Bag=False,
            Scarf=False,
            Left_shoe=False,
            Right_shoe=False,
            Background=False,
            process_res=512,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            background="Alpha",
            background_color="#222222",
            images=get_value_at_index(imagecrop_112, 0),
        )
        maskcomposite_92 = maskcomposite_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(yolov8_person_nomask_90, 0),
            source=get_value_at_index(clothessegment_91, 1),
        )
        focuscropultra_94 = focuscropultra_node.crop_by_mask_v2(
            up_keep=0.2,
            down_keep=0.2,
            right_keep=0.2,
            left_keep=0.2,
            crop_ratio="original",
            aspect_ratio="original",
            fit="fill",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1520,
            background_color="#000000",
            expand=0,
            blur_radius=0,
            image=get_value_at_index(imagecrop_112, 0),
            mask=get_value_at_index(maskcomposite_92, 0),
        )
        getimagesize_5 = getimagesize_node.execute(image=get_value_at_index(focuscropultra_94, 3))
        painterfluximageedit_44 = painterfluximageedit_node.encode(
            prompt=get_value_at_index(cr_text_48, 0),
            width=get_value_at_index(getimagesize_5, 0),
            height=get_value_at_index(getimagesize_5, 1),
            clip=get_value_at_index(self.cliploader_88, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
            image1=get_value_at_index(focuscropultra_94, 3),
            image1_mask=get_value_at_index(focuscropultra_94, 4),
        )
        layerutility_cropboxresolve_100 = layerutility_cropboxresolve_node.crop_box_resolve(
            crop_box=get_value_at_index(focuscropultra_94, 2)
        )
        lanpaint_ksampler_49 = lanpaint_ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            LanPaint_NumSteps=3,
            LanPaint_PromptMode="Image First",
            LanPaint_Info="LanPaint KSampler. For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!",
            Inpainting_mode="🖼️ Image Inpainting",
            model=get_value_at_index(self.unetloader_89, 0),
            positive=get_value_at_index(painterfluximageedit_44, 0),
            negative=get_value_at_index(painterfluximageedit_44, 1),
            latent_image=get_value_at_index(painterfluximageedit_44, 2),
        )
        vaedecode_4 = vaedecode_node.decode(
            samples=get_value_at_index(lanpaint_ksampler_49, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
        )
        jwimageresize_23 = jwimageresize_node.execute(
            height=get_value_at_index(layerutility_cropboxresolve_100, 3),
            width=get_value_at_index(layerutility_cropboxresolve_100, 2),
            interpolation_mode="bicubic",
            image=get_value_at_index(vaedecode_4, 0),
        )
        imagecompositemasked_19 = imagecompositemasked_node.EXECUTE_NORMALIZED(
            x=get_value_at_index(layerutility_cropboxresolve_100, 0),
            y=get_value_at_index(layerutility_cropboxresolve_100, 1),
            resize_source=False,
            destination=get_value_at_index(imagecrop_112, 0),
            source=get_value_at_index(jwimageresize_23, 0),
        )
        yolov8_person_nomask_30 = yolov8_person_nomask_node.yolov8_person_nomask(
            yolo_model="yolo11m-seg.pt",
            true_rate=0.1,
            img_ratio=0.6666666666666666,
            x_ratio=0.5,
            y_ratio=0.1,
            radius=100,
            blur_radius=0,
            back_image=get_value_at_index(imagecompositemasked_19, 0),
        )
        clothessegment_28 = clothessegment_node.segment_clothes(
            Hat=True,
            Hair=False,
            Face=False,
            Sunglasses=True,
            Upper_clothes=False,
            Skirt=False,
            Dress=False,
            Belt=False,
            Pants=False,
            Left_arm=False,
            Right_arm=False,
            Left_leg=False,
            Right_leg=False,
            Bag=False,
            Scarf=False,
            Left_shoe=False,
            Right_shoe=False,
            Background=False,
            process_res=512,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            background="Alpha",
            background_color="#222222",
            images=get_value_at_index(imagecompositemasked_19, 0),
        )
        maskcomposite_29 = maskcomposite_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(yolov8_person_nomask_30, 0),
            source=get_value_at_index(clothessegment_28, 1),
        )
        invertmask_27 = invertmask_node.EXECUTE_NORMALIZED(
            mask=get_value_at_index(maskcomposite_29, 0)
        )
        layerutility_imagescalebyaspectratio_v2_26 = (
            layerutility_imagescalebyaspectratio_v2_node.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="letterbox",
                method="lanczos",
                round_to_multiple="8",
                scale_to_side="longest",
                scale_to_length=1520,
                background_color="#000000",
                image=get_value_at_index(imagecompositemasked_19, 0),
                mask=get_value_at_index(invertmask_27, 0),
            )
        )
        getimagesize_12 = getimagesize_node.execute(
            image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_26, 0)
        )
        painterfluximageedit_46 = painterfluximageedit_node.encode(
            prompt=get_value_at_index(cr_text_13, 0),
            width=get_value_at_index(getimagesize_12, 0),
            height=get_value_at_index(getimagesize_12, 1),
            clip=get_value_at_index(self.cliploader_88, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
            image1=get_value_at_index(layerutility_imagescalebyaspectratio_v2_26, 0),
            image1_mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_26, 1),
        )
        lanpaint_ksampler_51 = lanpaint_ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            LanPaint_NumSteps=3,
            LanPaint_PromptMode="Image First",
            LanPaint_Info="LanPaint KSampler. For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!",
            Inpainting_mode="🖼️ Image Inpainting",
            model=get_value_at_index(self.unetloader_89, 0),
            positive=get_value_at_index(painterfluximageedit_46, 0),
            negative=get_value_at_index(painterfluximageedit_46, 1),
            latent_image=get_value_at_index(painterfluximageedit_46, 2),
        )
        vaedecode_36 = vaedecode_node.decode(
            samples=get_value_at_index(lanpaint_ksampler_51, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
        )
        imagemaskscaleasv2_37 = imagemaskscaleasv2_node.image_mask_scale_as_v2(
            fit="letterbox",
            method="lanczos",
            background_color="#FFFFFF",
            scale_as=get_value_at_index(imagecompositemasked_19, 0),
            image=get_value_at_index(vaedecode_36, 0),
        )
        res = tensor2pil(get_value_at_index(imagemaskscaleasv2_37, 0))

        # 尽量释放中间大 tensor（依赖作用域自动 GC）
        try:
            del (
                cr_text_13,
                cr_text_48,
                loadimage_78,
                loadimage_79,
                getimagesize_55,
                getimagesize_69,
                impactcompare_64,
                layerutility_numbercalculator_60,
                imagescalebyaspectratiov2_67,
                imagescalebyaspectratiov2_65,
                easy_ifelse_73,
                rmbg_62,
                imagecompositemasked_57,
                imagecrop_112,
                yolov8_person_nomask_90,
                clothessegment_91,
                maskcomposite_92,
                focuscropultra_94,
                getimagesize_5,
                painterfluximageedit_44,
                layerutility_cropboxresolve_100,
                lanpaint_ksampler_49,
                vaedecode_4,
                jwimageresize_23,
                imagecompositemasked_19,
                yolov8_person_nomask_30,
                clothessegment_28,
                maskcomposite_29,
                invertmask_27,
                layerutility_imagescalebyaspectratio_v2_26,
                getimagesize_12,
                painterfluximageedit_46,
                lanpaint_ksampler_51,
                vaedecode_36,
                imagemaskscaleasv2_37,
            )
        except Exception:
            pass

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return res


# if __name__ == "__main__":
#     flux2_cb_processor = fflux2_klein_change_bg()
#     init_image = Image.open("../my_images/0-0.jpg")
#     bg_image = Image.open("../my_images/0-1.jpg")
#     res_image = flux2_cb_processor.forward(init_image, bg_image)
#     res_image.save("../my_images/0-2_res_flux.jpg")
