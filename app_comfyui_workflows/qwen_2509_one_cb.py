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
        print(f"{name} found: {path_name}")
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
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")
    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
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


import_custom_nodes()
from nodes import NODE_CLASS_MAPPINGS

# 节点实例（与 flux2_klein_one_cb 风格一致，模块级缓存）
getimagesize_node = NODE_CLASS_MAPPINGS["GetImageSize+"]()
impactcompare_node = NODE_CLASS_MAPPINGS["ImpactCompare"]()
layerutility_numbercalculator_node = NODE_CLASS_MAPPINGS["LayerUtility: NumberCalculator"]()
imagescalebyaspectratiov2_node = NODE_CLASS_MAPPINGS["ImageScaleByAspectRatioV2"]()
easy_ifelse_node = NODE_CLASS_MAPPINGS["easy ifElse"]()
rmbg_node = NODE_CLASS_MAPPINGS["RMBG"]()
imagecompositemasked_node = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
imagecrop_node = NODE_CLASS_MAPPINGS["ImageCrop"]()
vaeencode_node = NODE_CLASS_MAPPINGS["VAEEncode"]()
text_multiline_node = NODE_CLASS_MAPPINGS["Text Multiline"]()
textencodeqwenimageeditplus_node = NODE_CLASS_MAPPINGS["TextEncodeQwenImageEditPlus"]()
conditioningzeroout_node = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
ksampler_node = NODE_CLASS_MAPPINGS["KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()


class qwen_2509_one_cb:
    def __init__(self):
        # 预加载模型，风格对齐 flux2_klein_one_cb
        self.vaeloader_86 = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name="qwen_image_vae.safetensors")
        self.unetloader_89 = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name="qwen/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            weight_dtype="fp8_e4m3fn",
        )
        self.loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        self.loraloadermodelonly_22 = self.loraloadermodelonly.load_lora_model_only(
            lora_name="qwen/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors",
            strength_model=1,
            model=get_value_at_index(self.unetloader_89, 0),
        )
        self.modelsamplingauraflow = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
        self.modelsamplingauraflow_54 = self.modelsamplingauraflow.patch_aura(
            shift=3, model=get_value_at_index(self.loraloadermodelonly_22, 0)
        )
        self.cfgnorm = NODE_CLASS_MAPPINGS["CFGNorm"]()
        self.cfgnorm_45 = self.cfgnorm.EXECUTE_NORMALIZED(
            strength=1, model=get_value_at_index(self.modelsamplingauraflow_54, 0)
        )
        self.loraloadermodelonly_67 = self.loraloadermodelonly.load_lora_model_only(
            lora_name="qwen/光影渲染产品溶图Qwen-Edit_2509.safetensors",
            strength_model=1,
            model=get_value_at_index(self.cfgnorm_45, 0),
        )
        self.cliploader_88 = NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(
            clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            type="stable_diffusion",
            device="default",
        )
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self, init_image: Image.Image, bg_image: Image.Image) -> Image.Image:
        """
        前景图 init_image + 背景图 bg_image -> Qwen 2509 流水线换背景后的 PIL.Image
        """
        loadimage_49 = self.loadimage.load_image(image=init_image)
        loadimage_50 = self.loadimage.load_image(image=bg_image)

        getimagesize_9 = getimagesize_node.execute(image=get_value_at_index(loadimage_49, 0))
        getimagesize_29 = getimagesize_node.execute(image=get_value_at_index(loadimage_50, 0))

        impactcompare_20 = impactcompare_node.doit(
            cmp="a <= b",
            a=get_value_at_index(getimagesize_29, 0),
            b=get_value_at_index(getimagesize_29, 1),
        )
        layerutility_numbercalculator_16 = (
            layerutility_numbercalculator_node.number_calculator_node(
                operator="max",
                a=get_value_at_index(getimagesize_9, 0),
                b=get_value_at_index(getimagesize_9, 1),
            )
        )
        imagescalebyaspectratiov2_27 = imagescalebyaspectratiov2_node.image_scale_by_aspect_ratio(
            aspect_ratio="original",
            proportional_width=1,
            proportional_height=1,
            fit="letterbox",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=get_value_at_index(layerutility_numbercalculator_16, 0),
            background_color="#000000",
            image=get_value_at_index(loadimage_50, 0),
        )
        imagescalebyaspectratiov2_21 = imagescalebyaspectratiov2_node.image_scale_by_aspect_ratio(
            aspect_ratio="original",
            proportional_width=1,
            proportional_height=1,
            fit="letterbox",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="shortest",
            scale_to_length=get_value_at_index(layerutility_numbercalculator_16, 0),
            background_color="#000000",
            image=get_value_at_index(loadimage_50, 0),
        )
        easy_ifelse_47 = easy_ifelse_node.execute(
            boolean=get_value_at_index(impactcompare_20, 0),
            on_true=get_value_at_index(imagescalebyaspectratiov2_27, 0),
            on_false=get_value_at_index(imagescalebyaspectratiov2_21, 0),
        )
        rmbg_18 = rmbg_node.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Alpha",
            background_color="#222222",
            image=get_value_at_index(loadimage_49, 0),
        )
        imagecompositemasked_12 = imagecompositemasked_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(easy_ifelse_47, 0),
            source=get_value_at_index(rmbg_18, 0),
            mask=get_value_at_index(rmbg_18, 1),
        )
        imagecrop_13 = imagecrop_node.EXECUTE_NORMALIZED(
            width=get_value_at_index(getimagesize_9, 0),
            height=get_value_at_index(getimagesize_9, 1),
            x=0,
            y=0,
            image=get_value_at_index(imagecompositemasked_12, 0),
        )
        vaeencode_77 = vaeencode_node.encode(
            pixels=get_value_at_index(imagecrop_13, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
        )
        text_multiline_84 = text_multiline_node.text_multiline(
            text="Give your subject natural light and shadow effects, blending it seamlessly with its surroundings while maintaining a consistent perspective between the subject and the background."
        )
        textencodeqwenimageeditplus_62 = textencodeqwenimageeditplus_node.EXECUTE_NORMALIZED(
            prompt=get_value_at_index(text_multiline_84, 0),
            clip=get_value_at_index(self.cliploader_88, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
            image1=get_value_at_index(imagecrop_13, 0),
        )
        conditioningzeroout_71 = conditioningzeroout_node.zero_out(
            conditioning=get_value_at_index(textencodeqwenimageeditplus_62, 0)
        )
        ksampler_74 = ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            model=get_value_at_index(self.loraloadermodelonly_67, 0),
            positive=get_value_at_index(textencodeqwenimageeditplus_62, 0),
            negative=get_value_at_index(conditioningzeroout_71, 0),
            latent_image=get_value_at_index(vaeencode_77, 0),
        )
        vaedecode_73 = vaedecode_node.decode(
            samples=get_value_at_index(ksampler_74, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
        )
        return tensor2pil(get_value_at_index(vaedecode_73, 0))


# if __name__ == "__main__":
#     qwen_cb_processor = qwen_2509_one_cb()
#     init_image = Image.open("../my_images/0-0.jpg")
#     bg_image = Image.open("../my_images/0-1.jpg")
#     res_image = qwen_cb_processor.forward(init_image, bg_image)
#     res_image.save("../my_images/0-2_res_qwen.jpg")
