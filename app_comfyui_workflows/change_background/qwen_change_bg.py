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
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        pass
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
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

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())
    os.environ["COMFYUI_NODES_LOADED"] = "1"





import_custom_nodes()
from nodes import NODE_CLASS_MAPPINGS

# 节点实例（与 flux2-klein_change_bg 风格一致，模块级缓存）
vaeloader_node = NODE_CLASS_MAPPINGS["VAELoader"]()
unetloader_node = NODE_CLASS_MAPPINGS["UNETLoader"]()
loraloadermodelonly_node = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
modelsamplingauraflow_node = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
cfgnorm_node = NODE_CLASS_MAPPINGS["CFGNorm"]()
cliploader_node = NODE_CLASS_MAPPINGS["CLIPLoader"]()
unetloadergguf_node = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
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
vaeencode_node = NODE_CLASS_MAPPINGS["VAEEncode"]()
text_multiline_node = NODE_CLASS_MAPPINGS["Text Multiline"]()
layerutility_cropboxresolve_node = NODE_CLASS_MAPPINGS["LayerUtility: CropBoxResolve"]()
textencodeqwenimageeditplus_node = NODE_CLASS_MAPPINGS["TextEncodeQwenImageEditPlus"]()
conditioningzeroout_node = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
setlatentnoisemask_node = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
ksampler_node = NODE_CLASS_MAPPINGS["KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
jwimageresize_node = NODE_CLASS_MAPPINGS["JWImageResize"]()
invertmask_node = NODE_CLASS_MAPPINGS["InvertMask"]()
layerutility_imagescalebyaspectratio_v2_node = NODE_CLASS_MAPPINGS["LayerUtility: ImageScaleByAspectRatio V2"]()
imagemaskscaleasv2_node = NODE_CLASS_MAPPINGS["ImageMaskScaleAsV2"]()


class qwen_change_bg:
    def __init__(self):
        # 预加载较重的模型，风格对齐 flux2-klein_change_bg / hand_beauty_full
        self.vaeloader_7 = vaeloader_node.load_vae(vae_name="qwen_image_vae.safetensors")
        self.unetloader_101 = unetloader_node.load_unet(
            unet_name="qwen/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            weight_dtype="fp8_e4m3fn",
        )
        self.loraloadermodelonly_26 = loraloadermodelonly_node.load_lora_model_only(
            lora_name="qwen/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors",
            strength_model=1,
            model=get_value_at_index(self.unetloader_101, 0),
        )
        self.modelsamplingauraflow_104 = modelsamplingauraflow_node.patch_aura(
            shift=3, model=get_value_at_index(self.loraloadermodelonly_26, 0)
        )
        self.cfgnorm_52 = cfgnorm_node.EXECUTE_NORMALIZED(
            strength=1, model=get_value_at_index(self.modelsamplingauraflow_104, 0)
        )
        self.loraloadermodelonly_9 = loraloadermodelonly_node.load_lora_model_only(
            lora_name="qwen/光影渲染产品溶图Qwen-Edit_2509.safetensors",
            strength_model=1,
            model=get_value_at_index(self.cfgnorm_52, 0),
        )
        self.loraloadermodelonly_68 = loraloadermodelonly_node.load_lora_model_only(
            lora_name="qwen/光影渲染产品溶图Qwen-Edit_2509.safetensors",
            strength_model=1,
            model=get_value_at_index(self.cfgnorm_52, 0),
        )
        self.cliploader_103 = cliploader_node.load_clip(
            clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            type="stable_diffusion",
            device="default",
        )
        self.unetloadergguf_100 = unetloadergguf_node.load_unet(
            unet_name="Qwen-Image-Edit-2509-Q8_0.gguf"
        )

        # 支持 PIL.Image 的 LoadImage 节点
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self, init_image: Image.Image, bg_image: Image.Image) -> Image.Image:
        """
        前景图 init_image + 背景图 bg_image -> Qwen 流水线换背景后的 PIL.Image
        """
        # 使用支持 PIL 的 LoadImage
        loadimage_98 = self.loadimage.load_image(image=init_image)
        loadimage_99 = self.loadimage.load_image(image=bg_image)

        getimagesize_10 = getimagesize_node.execute(
            image=get_value_at_index(loadimage_98, 0)
        )
        getimagesize_33 = getimagesize_node.execute(
            image=get_value_at_index(loadimage_99, 0)
        )

        impactcompare_24 = impactcompare_node.doit(
            cmp="a <= b",
            a=get_value_at_index(getimagesize_33, 0),
            b=get_value_at_index(getimagesize_33, 1),
        )

        layerutility_numbercalculator_20 = (
            layerutility_numbercalculator_node.number_calculator_node(
                operator="max",
                a=get_value_at_index(getimagesize_10, 0),
                b=get_value_at_index(getimagesize_10, 1),
            )
        )

        imagescalebyaspectratiov2_31 = (
            imagescalebyaspectratiov2_node.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="letterbox",
                method="lanczos",
                round_to_multiple="8",
                scale_to_side="longest",
                scale_to_length=get_value_at_index(layerutility_numbercalculator_20, 0),
                background_color="#000000",
                image=get_value_at_index(loadimage_99, 0),
            )
        )
        imagescalebyaspectratiov2_25 = (
            imagescalebyaspectratiov2_node.image_scale_by_aspect_ratio(
                aspect_ratio="original",
                proportional_width=1,
                proportional_height=1,
                fit="letterbox",
                method="lanczos",
                round_to_multiple="8",
                scale_to_side="shortest",
                scale_to_length=get_value_at_index(layerutility_numbercalculator_20, 0),
                background_color="#000000",
                image=get_value_at_index(loadimage_99, 0),
            )
        )

        easy_ifelse_60 = easy_ifelse_node.execute(
            boolean=get_value_at_index(impactcompare_24, 0),
            on_true=get_value_at_index(imagescalebyaspectratiov2_31, 0),
            on_false=get_value_at_index(imagescalebyaspectratiov2_25, 0),
        )

        rmbg_22 = rmbg_node.process_image(
            model="RMBG-2.0",
            sensitivity=1,
            process_res=1024,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            refine_foreground=False,
            background="Alpha",
            background_color="#222222",
            image=get_value_at_index(loadimage_98, 0),
        )

        imagecompositemasked_16 = imagecompositemasked_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(easy_ifelse_60, 0),
            source=get_value_at_index(rmbg_22, 0),
            mask=get_value_at_index(rmbg_22, 1),
        )

        imagecrop_17 = imagecrop_node.EXECUTE_NORMALIZED(
            width=get_value_at_index(getimagesize_10, 0),
            height=get_value_at_index(getimagesize_10, 1),
            x=0,
            y=0,
            image=get_value_at_index(imagecompositemasked_16, 0),
        )

        yolov8_person_nomask_112 = yolov8_person_nomask_node.yolov8_person_nomask(
            yolo_model="yolo11m-seg.pt",
            true_rate=0.1,
            img_ratio=0.6666666666666666,
            x_ratio=0.5,
            y_ratio=0.1,
            radius=100,
            blur_radius=0,
            back_image=get_value_at_index(imagecrop_17, 0),
        )

        clothessegment_113 = clothessegment_node.segment_clothes(
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
            images=get_value_at_index(imagecrop_17, 0),
        )

        maskcomposite_114 = maskcomposite_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(yolov8_person_nomask_112, 0),
            source=get_value_at_index(clothessegment_113, 1),
        )

        focuscropultra_116 = focuscropultra_node.crop_by_mask_v2(
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
            image=get_value_at_index(imagecrop_17, 0),
            mask=get_value_at_index(maskcomposite_114, 0),
        )

        vaeencode_57 = vaeencode_node.encode(
            pixels=get_value_at_index(focuscropultra_116, 3),
            vae=get_value_at_index(self.vaeloader_7, 0),
        )

        text_multiline_71 = text_multiline_node.text_multiline(
            text="Give your subject natural light and shadow effects, blending it seamlessly with its surroundings while maintaining a consistent perspective between the subject and the background."
        )

        layerutility_cropboxresolve_122 = layerutility_cropboxresolve_node.crop_box_resolve(
            crop_box=get_value_at_index(focuscropultra_116, 2)
        )

        text_multiline_107 = text_multiline_node.text_multiline(
            text="Give your subject natural light and shadow effects, blending it seamlessly with its surroundings while maintaining a consistent perspective between the subject and the background."
        )

        textencodeqwenimageeditplus_1 = textencodeqwenimageeditplus_node.EXECUTE_NORMALIZED(
            prompt=get_value_at_index(text_multiline_107, 0),
            clip=get_value_at_index(self.cliploader_103, 0),
            vae=get_value_at_index(self.vaeloader_7, 0),
            image1=get_value_at_index(focuscropultra_116, 3),
        )

        conditioningzeroout_34 = conditioningzeroout_node.zero_out(
            conditioning=get_value_at_index(textencodeqwenimageeditplus_1, 0)
        )

        setlatentnoisemask_56 = setlatentnoisemask_node.set_mask(
            samples=get_value_at_index(vaeencode_57, 0),
            mask=get_value_at_index(focuscropultra_116, 4),
        )

        ksampler_254 = ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            model=get_value_at_index(self.loraloadermodelonly_9, 0),
            positive=get_value_at_index(textencodeqwenimageeditplus_1, 0),
            negative=get_value_at_index(conditioningzeroout_34, 0),
            latent_image=get_value_at_index(setlatentnoisemask_56, 0),
        )

        vaedecode_51 = vaedecode_node.decode(
            samples=get_value_at_index(ksampler_254, 0),
            vae=get_value_at_index(self.vaeloader_7, 0),
        )

        jwimageresize_87 = jwimageresize_node.execute(
            height=get_value_at_index(layerutility_cropboxresolve_122, 3),
            width=get_value_at_index(layerutility_cropboxresolve_122, 2),
            interpolation_mode="bicubic",
            image=get_value_at_index(vaedecode_51, 0),
        )

        imagecompositemasked_128 = imagecompositemasked_node.EXECUTE_NORMALIZED(
            x=get_value_at_index(layerutility_cropboxresolve_122, 0),
            y=get_value_at_index(layerutility_cropboxresolve_122, 1),
            resize_source=False,
            destination=get_value_at_index(imagecrop_17, 0),
            source=get_value_at_index(jwimageresize_87, 0),
        )

        yolov8_person_nomask_96 = yolov8_person_nomask_node.yolov8_person_nomask(
            yolo_model="yolo11m-seg.pt",
            true_rate=0.1,
            img_ratio=0.6666666666666666,
            x_ratio=0.5,
            y_ratio=0.1,
            radius=100,
            blur_radius=0,
            back_image=get_value_at_index(imagecompositemasked_128, 0),
        )

        clothessegment_140 = clothessegment_node.segment_clothes(
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
            images=get_value_at_index(imagecompositemasked_128, 0),
        )

        maskcomposite_97 = maskcomposite_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(yolov8_person_nomask_96, 0),
            source=get_value_at_index(clothessegment_140, 1),
        )

        invertmask_92 = invertmask_node.EXECUTE_NORMALIZED(
            mask=get_value_at_index(maskcomposite_97, 0)
        )

        layerutility_imagescalebyaspectratio_v2_91 = (
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
                image=get_value_at_index(imagecompositemasked_128, 0),
                mask=get_value_at_index(invertmask_92, 0),
            )
        )

        vaeencode_85 = vaeencode_node.encode(
            pixels=get_value_at_index(layerutility_imagescalebyaspectratio_v2_91, 0),
            vae=get_value_at_index(self.vaeloader_7, 0),
        )

        # 第二阶段 Qwen-Edit inpaint
        textencodeqwenimageeditplus_74 = (
            textencodeqwenimageeditplus_node.EXECUTE_NORMALIZED(
                prompt=get_value_at_index(text_multiline_71, 0),
                clip=get_value_at_index(self.cliploader_103, 0),
                vae=get_value_at_index(self.vaeloader_7, 0),
                image1=get_value_at_index(
                    layerutility_imagescalebyaspectratio_v2_91, 0
                ),
            )
        )

        conditioningzeroout_69 = conditioningzeroout_node.zero_out(
            conditioning=get_value_at_index(textencodeqwenimageeditplus_74, 0)
        )

        setlatentnoisemask_86 = setlatentnoisemask_node.set_mask(
            samples=get_value_at_index(vaeencode_85, 0),
            mask=get_value_at_index(layerutility_imagescalebyaspectratio_v2_91, 1),
        )

        ksampler_255 = ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=8,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            model=get_value_at_index(self.loraloadermodelonly_68, 0),
            positive=get_value_at_index(textencodeqwenimageeditplus_74, 0),
            negative=get_value_at_index(conditioningzeroout_69, 0),
            latent_image=get_value_at_index(setlatentnoisemask_86, 0),
        )

        vaedecode_76 = vaedecode_node.decode(
            samples=get_value_at_index(ksampler_255, 0),
            vae=get_value_at_index(self.vaeloader_7, 0),
        )

        imagemaskscaleasv2_81 = imagemaskscaleasv2_node.image_mask_scale_as_v2(
            fit="letterbox",
            method="lanczos",
            background_color="#FFFFFF",
            scale_as=get_value_at_index(imagecompositemasked_128, 0),
            image=get_value_at_index(vaedecode_76, 0),
        )

        # 返回最终结果（经第二阶段 Qwen 编辑后的图）
        pil_image=tensor2pil(get_value_at_index(imagemaskscaleasv2_81, 0))
        return pil_image

# if __name__=="__main__":
#     qwen_change_bg=qwen_change_bg()
#     init_image=Image.open("../my_images/0-0.jpg")
#     bg_image=Image.open("../my_images/0-1.jpg")
#     res_image=qwen_change_bg.forward(init_image,bg_image)
#     res_image.save("../my_images/0-2_res_qwen.jpg")
