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


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)


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
cliptextencode_node = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
textencodeqwenimageeditplus_node = NODE_CLASS_MAPPINGS["TextEncodeQwenImageEditPlus"]()
getimagesize_plus_node = NODE_CLASS_MAPPINGS["GetImageSize+"]()
emptylatentimage_node = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
ksampler_node = NODE_CLASS_MAPPINGS["KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()


class qwen_change_sight:
    def __init__(self):
        self.name = self.__class__.__name__

        # 预加载模型
        self.unetloader_21 = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name="qwen/qwen_image_edit_2511_bf16.safetensors",
            weight_dtype="default",
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        self.loraloadermodelonly_22 = loraloadermodelonly.load_lora_model_only(
            lora_name="qwen/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",
            strength_model=1,
            model=get_value_at_index(self.unetloader_21, 0),
        )

        self.loraloadermodelonly_23 = loraloadermodelonly.load_lora_model_only(
            lora_name="qwen/qwen-image-edit-2511-multiple-angles-lora.safetensors",
            strength_model=1,
            model=get_value_at_index(self.loraloadermodelonly_22, 0),
        )

        self.cliploader_24 = NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(
            clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            type="qwen_image",
            device="default",
        )

        self.vaeloader_25 = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name="qwen_image_vae.safetensors")

        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self, image, prompt):
        """
        前向处理流程，接收输入图像和提示词 -> 输出结果 PIL.Image
        """
        loadimage_30 = self.loadimage.load_image(image=image)

        text_multiline_33 = text_multiline_node.text_multiline(text=prompt)

        cliptextencode_36 = cliptextencode_node.encode(
            text="泛黄，AI感，不真实，丑陋，油腻的皮肤，异常的肢体，不协调的肢体",
            clip=get_value_at_index(self.cliploader_24, 0),
        )

        textencodeqwenimageeditplus_29 = textencodeqwenimageeditplus_node.EXECUTE_NORMALIZED(
            prompt=get_value_at_index(text_multiline_33, 0),
            clip=get_value_at_index(self.cliploader_24, 0),
            vae=get_value_at_index(self.vaeloader_25, 0),
            image1=get_value_at_index(loadimage_30, 0),
        )

        getimagesize_39 = getimagesize_plus_node.execute(
            image=get_value_at_index(loadimage_30, 0)
        )

        emptylatentimage_38 = emptylatentimage_node.generate(
            width=get_value_at_index(getimagesize_39, 0),
            height=get_value_at_index(getimagesize_39, 1),
            batch_size=1,
        )

        ksampler_34 = ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="beta",
            denoise=1,
            model=get_value_at_index(self.loraloadermodelonly_23, 0),
            positive=get_value_at_index(textencodeqwenimageeditplus_29, 0),
            negative=get_value_at_index(cliptextencode_36, 0),
            latent_image=get_value_at_index(emptylatentimage_38, 0),
        )

        vaedecode_40 = vaedecode_node.decode(
            samples=get_value_at_index(ksampler_34, 0),
            vae=get_value_at_index(self.vaeloader_25, 0),
        )

        res_image = tensor2pil(get_value_at_index(vaedecode_40, 0))

        return res_image

