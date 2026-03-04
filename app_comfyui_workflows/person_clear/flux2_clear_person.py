import os
import random
import sys
from typing import Sequence, Mapping, Any, Union, Tuple
import torch
import numpy as np
from PIL import Image
import types
from functools import wraps


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
        if comfyui_path not in sys.path:
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

# 节点实例（模块级缓存）
text_multiline_node = NODE_CLASS_MAPPINGS["Text Multiline"]()
imagetomask_node = NODE_CLASS_MAPPINGS["ImageToMask"]()
imagescalebyaspectratiov2_node = NODE_CLASS_MAPPINGS["ImageScaleByAspectRatioV2"]()
getimagesize_plus_node = NODE_CLASS_MAPPINGS["GetImageSize+"]()
layermask_maskgrow_node = NODE_CLASS_MAPPINGS["LayerMask: MaskGrow"]()
painterfluximageedit_node = NODE_CLASS_MAPPINGS["PainterFluxImageEdit"]()
getimagesize_node = NODE_CLASS_MAPPINGS["GetImageSize"]()
lanpaint_ksampler_node = NODE_CLASS_MAPPINGS["LanPaint_KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
imagescale_node = NODE_CLASS_MAPPINGS["ImageScale"]()


class flux2_clear_person:
    def __init__(self):
        self.name = self.__class__.__name__

        # 预加载模型，风格对齐
        self.unetloader_46 = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name="flux-2-klein-9b-fp8.safetensors", weight_dtype="default"
        )
        self.cliploader_49 = NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(
            clip_name="split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
            type="flux2",
            device="default",
        )
        self.vaeloader_50 = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name="flux2-vae.safetensors")

        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self, image,mask_image):
        """
        前向处理流程，接收输入请求和时间戳 -> 输出结果字典
        """
        prompt = "消除图1中的人物"

        text_multiline_60 = text_multiline_node.text_multiline(text=prompt)
        loadimage_24 = self.loadimage.load_image(image=image)
        loadimage_61 = self.loadimage.load_image(image=mask_image)

        imagetomask_63 = imagetomask_node.EXECUTE_NORMALIZED(
            channel="red", image=get_value_at_index(loadimage_61, 0)
        )

        imagescalebyaspectratiov2_26 = imagescalebyaspectratiov2_node.image_scale_by_aspect_ratio(
            aspect_ratio="original",
            proportional_width=1,
            proportional_height=1,
            fit="letterbox",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1536,
            background_color="#000000",
            image=get_value_at_index(loadimage_24, 0),
            mask=get_value_at_index(imagetomask_63, 0),
        )

        getimagesize_59 = getimagesize_plus_node.execute(
            image=get_value_at_index(imagescalebyaspectratiov2_26, 0)
        )

        layermask_maskgrow_47 = layermask_maskgrow_node.mask_grow(
            invert_mask=False,
            grow=30,
            blur=0,
            mask=get_value_at_index(imagescalebyaspectratiov2_26, 1),
        )

        painterfluximageedit_51 = painterfluximageedit_node.encode(
            prompt=get_value_at_index(text_multiline_60, 0),
            width=get_value_at_index(getimagesize_59, 0),
            height=get_value_at_index(getimagesize_59, 1),
            clip=get_value_at_index(self.cliploader_49, 0),
            vae=get_value_at_index(self.vaeloader_50, 0),
            image1=get_value_at_index(imagescalebyaspectratiov2_26, 0),
            image1_mask=get_value_at_index(layermask_maskgrow_47, 0),
        )

        getimagesize_36 = getimagesize_node.EXECUTE_NORMALIZED(
            image=get_value_at_index(loadimage_24, 0), unique_id=9891662374490660455
        )

        lanpaint_ksampler_53 = lanpaint_ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=4,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            LanPaint_NumSteps=3,
            LanPaint_PromptMode="Image First",
            LanPaint_Info="LanPaint KSampler. For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!",
            Inpainting_mode="🖼️ Image Inpainting",
            model=get_value_at_index(self.unetloader_46, 0),
            positive=get_value_at_index(painterfluximageedit_51, 0),
            negative=get_value_at_index(painterfluximageedit_51, 1),
            latent_image=get_value_at_index(painterfluximageedit_51, 2),
        )

        vaedecode_54 = vaedecode_node.decode(
            samples=get_value_at_index(lanpaint_ksampler_53, 0),
            vae=get_value_at_index(self.vaeloader_50, 0),
        )

        imagescale_29 = imagescale_node.upscale(
            upscale_method="nearest-exact",
            width=get_value_at_index(getimagesize_36, 0),
            height=get_value_at_index(getimagesize_36, 1),
            crop="disabled",
            image=get_value_at_index(vaedecode_54, 0),
        )

        res_image = tensor2pil(get_value_at_index(imagescale_29, 0))
        
        return res_image


