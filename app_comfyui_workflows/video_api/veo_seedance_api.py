import os
import sys
import tempfile
import types
from functools import wraps
from typing import Sequence, Mapping, Any, Union
import numpy as np
import torch


def support_pil_image(original_method):
    """与 qwen_change_sight 一致：传入 PIL 时转成 tensor 并返回 LoadImage 同格式，否则走原逻辑。"""
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


def _pil_to_loadimage_path(pil_image):
    """将 PIL Image 保存到 ComfyUI input 目录并返回 LoadImage 可用的文件名（备用方案）。"""
    import folder_paths
    input_dir = folder_paths.get_input_directory()
    fd, path = tempfile.mkstemp(suffix=".png", dir=input_dir)
    os.close(fd)
    try:
        pil_image.save(path)
        return os.path.basename(path)
    except Exception:
        try:
            os.remove(path)
        except Exception:
            pass
        raise


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

# 节点实例（与 wan_vace_t2v 风格一致，模块级缓存）
loadimage_node = NODE_CLASS_MAPPINGS["LoadImage"]()
layerutility_scale_node = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
kim_video_t2v_node = NODE_CLASS_MAPPINGS["kim_video_T2V"]()
kim_video_i2v_node = NODE_CLASS_MAPPINGS["kim_video_I2V"]()
easy_showanything_node = NODE_CLASS_MAPPINGS["easy showAnything"]()


class VeoSeedanceAPI:
    def __init__(self):
        # 与 qwen_change_sight 一致：给 LoadImage 绑 support_pil_image，即可直接传 PIL
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self,image_1=None,image_2=None,prompt=None,api_key=None,model_select="veo3.1-fast",aspect_ratio="9:16",resolution="480p",duration=5):
        """
        双图 I2V + 可选 T2V：加载两张图，缩放后走 kim_video I2V，并可选跑 T2V。
        返回 (i2v_output, t2v_output)，均为对应节点 run 的返回值（可用 get_value_at_index(..., 0) 取视频等）。
        """

        if image_1 is None:
            kim_video_t2v_out = kim_video_t2v_node.run(
            api_key=api_key,
            model_select=model_select,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration,
            )

            return get_value_at_index(kim_video_t2v_out, 0)

        if image_2 is None:
            loadimage_1 = self.loadimage.load_image(image=image_1)
            scaled_1 = layerutility_scale_node.image_scale_by_aspect_ratio(
            aspect_ratio="original",
            proportional_width=1,
            proportional_height=1,
            fit="letterbox",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1024,
            background_color="#000000",
            image=get_value_at_index(loadimage_1, 0),
            )

            kim_video_i2v_out = kim_video_i2v_node.run(
            api_key=api_key,
            model_select=model_select,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration,
            image_1=get_value_at_index(scaled_1, 0),
            )

            return get_value_at_index(kim_video_i2v_out, 0)

        loadimage_1 = self.loadimage.load_image(image=image_1)
        loadimage_2 = self.loadimage.load_image(image=image_2)


        scaled_1 = layerutility_scale_node.image_scale_by_aspect_ratio(
            aspect_ratio="original",
            proportional_width=1,
            proportional_height=1,
            fit="letterbox",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1024,
            background_color="#000000",
            image=get_value_at_index(loadimage_1, 0),
        )
        scaled_2 = layerutility_scale_node.image_scale_by_aspect_ratio(
            aspect_ratio="original",
            proportional_width=1,
            proportional_height=1,
            fit="letterbox",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1024,
            background_color="#000000",
            image=get_value_at_index(loadimage_2, 0),
        )

        kim_video_i2v_out = kim_video_i2v_node.run(
            api_key=api_key,
            model_select=model_select,
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            duration=duration,
            image_1=get_value_at_index(scaled_1, 0),
            image_2=get_value_at_index(scaled_2, 0),
        )

        return get_value_at_index(kim_video_i2v_out, 0)
        


