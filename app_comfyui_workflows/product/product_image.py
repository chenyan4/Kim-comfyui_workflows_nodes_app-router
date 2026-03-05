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
ecommercepromptgenerator_node = NODE_CLASS_MAPPINGS["EcommercePromptGenerator"]()
easy_showanything_node = NODE_CLASS_MAPPINGS["easy showAnything"]()
synvownano2_i2i_node = NODE_CLASS_MAPPINGS["SynVowNano2_I2I"]()


class product_image:
    def __init__(self):
        self.name = self.__class__.__name__
        
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(
        self,
        image=None,
        product_type="耳机",
        prompt="主标题：MICKEY 无线头戴｜白绿潮范，声浪随行\n副标题：无线畅连 + 柔肤耳罩 标志性 米老鼠 标承包你的潮流听觉",
        design_style="科技感",
        scene_preference="混合（以使用场景为主）",
        output_language="中文 (Chinese)",
        api_key=None,
        model_select="gemini-3-pro-image-preview",
        size_mode="Match Image_1 (Smart Crop)",
        custom_w=768,
        custom_h=1344,
    ):
        """
        前向处理流程，接收产品图和参数 -> 输出生成的产品宣传图 PIL.Image
        """
        loadimage_8 = self.loadimage.load_image(image=image)

        ecommercepromptgenerator_61 = ecommercepromptgenerator_node.generate_prompts_with_vision(
            api_url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            api_key="7f27f95d8848453c8de1c2c2585002a3.0UNMcPLcxYF747IT",
            model_name="glm-4.6v-flash",
            product_type=product_type,
            selling_points=prompt,
            design_style=design_style,
            scene_preference=scene_preference,
            output_language=output_language,
            seed=random.randint(1, 99999),
            prompt_count=1,
            product_image=get_value_at_index(loadimage_8, 0),
        )

        easy_showanything_62 = easy_showanything_node.log_input(
            text="",
            anything=get_value_at_index(ecommercepromptgenerator_61, 0),
            unique_id=random.randint(1, 2**64),
        )

        synvownano2_i2i_135 = synvownano2_i2i_node.run(
            api_source="T8Star (ai.t8star.cn)",
            api_key=api_key,
            model_select=model_select,
            prompt=get_value_at_index(easy_showanything_62, 0),
            size_mode=size_mode,
            custom_w=custom_w,
            custom_h=custom_h,
            count=1,
            seed=random.randint(1, 2**64),
            image_1=get_value_at_index(loadimage_8, 0),
        )

        return tensor2pil(get_value_at_index(synvownano2_i2i_135, 0))

