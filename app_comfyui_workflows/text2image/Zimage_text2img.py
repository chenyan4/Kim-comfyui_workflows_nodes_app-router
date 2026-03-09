import os
import random
import sys
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

# 节点实例（与 flux2_klein_one_cb 风格一致，模块级缓存）
cliptextencode_node = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
emptylatentimage_node = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
fluxguidance_node = NODE_CLASS_MAPPINGS["FluxGuidance"]()
conditioningzeroout_node = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
ksampler_node = NODE_CLASS_MAPPINGS["KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()


class Zimage_text2img:
    def __init__(self):
        # 预加载模型，风格对齐 flux2_klein_one_cb
        self.vaeloader_86 = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name="ae.safetensors")
        self.cliploader_88 = NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(
            clip_name="qwen_3_4b.safetensors",
            type="stable_diffusion",
            device="default",
        )
        self.unetloader_89 = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name="Z-image/z_image_turbo_bf16.safetensors", weight_dtype="default"
        )

    @torch.inference_mode()
    def forward(self, prompt: str, width: int = 1080, height: int = 1920) -> Image.Image:
        """
        文本 prompt -> ZImage 文生图后的 PIL.Image
        """
        cliptextencode_5 = cliptextencode_node.encode(
            text=prompt,
            clip=get_value_at_index(self.cliploader_88, 0),
        )
        emptylatentimage_7 = emptylatentimage_node.generate(
            width=width, height=height, batch_size=1
        )
        fluxguidance_57 = fluxguidance_node.EXECUTE_NORMALIZED(
            guidance=50, conditioning=get_value_at_index(cliptextencode_5, 0)
        )
        conditioningzeroout_6 = conditioningzeroout_node.zero_out(
            conditioning=get_value_at_index(cliptextencode_5, 0)
        )
        ksampler_4 = ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=10,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            model=get_value_at_index(self.unetloader_89, 0),
            positive=get_value_at_index(fluxguidance_57, 0),
            negative=get_value_at_index(conditioningzeroout_6, 0),
            latent_image=get_value_at_index(emptylatentimage_7, 0),
        )
        vaedecode_8 = vaedecode_node.decode(
            samples=get_value_at_index(ksampler_4, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
        )
        return tensor2pil(get_value_at_index(vaedecode_8, 0))


# if __name__ == "__main__":
#     text2img_processor = Zimage_text2img()
#     prompt = "外貌：​ 一位年轻的东亚女性，面容清秀..."
#     res_image = text2img_processor.forward(prompt, width=1080, height=1920)
#     res_image.save("../my_images/zimage_text2img_res.jpg")
