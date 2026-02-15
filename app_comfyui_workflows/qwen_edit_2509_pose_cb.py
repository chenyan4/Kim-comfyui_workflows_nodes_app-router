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
layerutility_imagescalebyaspectratio_v2_node = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
rmbg_node = NODE_CLASS_MAPPINGS["RMBG"]()
focuscropultra_node = NODE_CLASS_MAPPINGS["FocusCropUltra"]()
dwpreprocessor_node = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
layerutility_imagemaskscaleas_node = NODE_CLASS_MAPPINGS["LayerUtility: ImageMaskScaleAs"]()
layerutility_imageblend_node = NODE_CLASS_MAPPINGS["LayerUtility: ImageBlend"]()
textencodeqwenimageeditplusadvance_lrzjason_node = NODE_CLASS_MAPPINGS["TextEncodeQwenImageEditPlusAdvance_lrzjason"]()
conditioningzeroout_node = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
ksampler_node = NODE_CLASS_MAPPINGS["KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
cropboxresolve_node = NODE_CLASS_MAPPINGS["CropBoxResolve"]()
imagescale_node = NODE_CLASS_MAPPINGS["ImageScale"]()
layerutility_restorecropbox_node = NODE_CLASS_MAPPINGS["LayerUtility: RestoreCropBox"]()


class qwen_edit_2509_pose_cb:
    def __init__(self):
        # 预加载模型，风格对齐 flux2_klein_one_cb
        self.cliploader_88 = NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(
            clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            type="qwen_image",
            device="default",
        )
        self.vaeloader_86 = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name="qwen_image_vae.safetensors")
        self.unetloadergguf = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
        self.unetloadergguf_89 = self.unetloadergguf.load_unet(
            unet_name="Qwen-Image-Edit-2509-Q8_0.gguf"
        )
        self.loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        self.loraloadermodelonly_22 = self.loraloadermodelonly.load_lora_model_only(
            lora_name="qwen/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors",
            strength_model=1,
            model=get_value_at_index(self.unetloadergguf_89, 0),
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
            lora_name="qwen/溶图.safetensors",
            strength_model=0.7,
            model=get_value_at_index(self.cfgnorm_45, 0),
        )
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self, person_image: Image.Image, pose_image: Image.Image) -> Image.Image:
        """
        人物图 person_image + 姿态参考图 pose_image -> Qwen 2509 姿态迁移后的 PIL.Image
        """
        loadimage_569 = self.loadimage.load_image(image=person_image)
        loadimage_608 = self.loadimage.load_image(image=pose_image)

        layerutility_imagescalebyaspectratio_v2_574 = (
            layerutility_imagescalebyaspectratio_v2_node.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="letterbox",
                method="lanczos",
                round_to_multiple="8",
                scale_to_side="longest",
                scale_to_length=2048,
                background_color="#000000",
                image=get_value_at_index(loadimage_608, 0),
            )
        )
        rmbg_1289 = rmbg_node.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Alpha",
            background_color="#222222",
            image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_574, 0),
        )
        focuscropultra_1291 = focuscropultra_node.crop_by_mask_v2(
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
            scale_to_length=1536,
            background_color="#000000",
            expand=0,
            blur_radius=0,
            image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_574, 0),
            mask=get_value_at_index(rmbg_1289, 1),
        )
        dwpreprocessor_1119 = dwpreprocessor_node.estimate_pose(
            detect_hand="enable",
            detect_body="enable",
            detect_face="enable",
            resolution=512,
            bbox_detector="yolox_l.onnx",
            pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
            scale_stick_for_xinsr_cn="disable",
            image=get_value_at_index(focuscropultra_1291, 3),
        )
        layerutility_imagemaskscaleas_1211 = (
            layerutility_imagemaskscaleas_node.image_mask_scale_as(
                fit="letterbox",
                method="lanczos",
                scale_as=get_value_at_index(focuscropultra_1291, 3),
                image=get_value_at_index(dwpreprocessor_1119, 0),
            )
        )
        layerutility_imageblend_600 = layerutility_imageblend_node.image_blend(
            invert_mask=True,
            blend_mode="normal",
            opacity=100,
            background_image=get_value_at_index(layerutility_imagemaskscaleas_1211, 0),
            layer_image=get_value_at_index(focuscropultra_1291, 3),
            layer_mask=get_value_at_index(focuscropultra_1291, 4),
        )
        rmbg_1216 = rmbg_node.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Color",
            background_color="#ffffff",
            image=get_value_at_index(loadimage_569, 0),
        )
        focuscropultra_1290 = focuscropultra_node.crop_by_mask_v2(
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
            scale_to_length=1536,
            background_color="#000000",
            expand=0,
            blur_radius=0,
            image=get_value_at_index(rmbg_1216, 0),
            mask=get_value_at_index(rmbg_1216, 1),
        )
        textencodeqwenimageeditplusadvance_lrzjason_582 = (
            textencodeqwenimageeditplusadvance_lrzjason_node.encode(
                prompt="将图1人物替换为图2人物，解决图、修正产品视角和光影净化产品背景",
                target_size=1344,
                target_vl_size=384,
                upscale_method="lanczos",
                crop_method="disabled",
                instruction="Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.",
                clip=get_value_at_index(self.cliploader_88, 0),
                vae=get_value_at_index(self.vaeloader_86, 0),
                vl_resize_image1=get_value_at_index(layerutility_imageblend_600, 0),
                vl_resize_image2=get_value_at_index(focuscropultra_1290, 3),
            )
        )
        conditioningzeroout_581 = conditioningzeroout_node.zero_out(
            conditioning=get_value_at_index(textencodeqwenimageeditplusadvance_lrzjason_582, 0)
        )
        ksampler_563 = ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="beta",
            denoise=1,
            model=get_value_at_index(self.loraloadermodelonly_67, 0),
            positive=get_value_at_index(textencodeqwenimageeditplusadvance_lrzjason_582, 0),
            negative=get_value_at_index(conditioningzeroout_581, 0),
            latent_image=get_value_at_index(textencodeqwenimageeditplusadvance_lrzjason_582, 1),
        )
        vaedecode_564 = vaedecode_node.decode(
            samples=get_value_at_index(ksampler_563, 0),
            vae=get_value_at_index(self.vaeloader_86, 0),
        )
        cropboxresolve_1275 = cropboxresolve_node.crop_box_resolve(
            crop_box=get_value_at_index(focuscropultra_1291, 2)
        )
        imagescale_1282 = imagescale_node.upscale(
            upscale_method="nearest-exact",
            width=get_value_at_index(cropboxresolve_1275, 2),
            height=get_value_at_index(cropboxresolve_1275, 3),
            crop="disabled",
            image=get_value_at_index(vaedecode_564, 0),
        )
        layerutility_restorecropbox_1229 = layerutility_restorecropbox_node.restore_crop_box(
            invert_mask=False,
            background_image=get_value_at_index(layerutility_imagescalebyaspectratio_v2_574, 0),
            croped_image=get_value_at_index(imagescale_1282, 0),
            crop_box=get_value_at_index(focuscropultra_1291, 2),
            croped_mask=get_value_at_index(focuscropultra_1291, 1),
        )
        return tensor2pil(get_value_at_index(layerutility_restorecropbox_1229, 0))


# if __name__ == "__main__":
#     pose_cb_processor = qwen_edit_2509_pose_cb()
#     person_image = Image.open("../my_images/person.jpg")
#     pose_image = Image.open("../my_images/pose.jpg")
#     res_image = pose_cb_processor.forward(person_image, pose_image)
#     res_image.save("../my_images/pose_transfer_res.jpg")
