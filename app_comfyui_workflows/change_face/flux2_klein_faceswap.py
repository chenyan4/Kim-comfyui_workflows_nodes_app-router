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

# 节点实例（与 flux2_klein_change_bg 风格一致，模块级缓存）
cr_text_node = NODE_CLASS_MAPPINGS["CR Text"]()
loadimage_node = NODE_CLASS_MAPPINGS["LoadImage"]()
facedetector_node = NODE_CLASS_MAPPINGS["FaceDetector"]()
focuscropultra_node = NODE_CLASS_MAPPINGS["FocusCropUltra"]()
getimagesize_node = NODE_CLASS_MAPPINGS["GetImageSize+"]()
emptyimagepro_node = NODE_CLASS_MAPPINGS["EmptyImagePro"]()
layermask_personmaskultra_node = NODE_CLASS_MAPPINGS["LayerMask: PersonMaskUltra"]()
facesegment_node = NODE_CLASS_MAPPINGS["FaceSegment"]()
maskcomposite_node = NODE_CLASS_MAPPINGS["MaskComposite"]()
growmask_node = NODE_CLASS_MAPPINGS["GrowMask"]()
imagecompositemasked_node = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
painterfluximageedit_node = NODE_CLASS_MAPPINGS["PainterFluxImageEdit"]()
lanpaint_ksampler_node = NODE_CLASS_MAPPINGS["LanPaint_KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
ksampler_node = NODE_CLASS_MAPPINGS["KSampler"]()
colormatch_node = NODE_CLASS_MAPPINGS["ColorMatch"]()
imagemaskscaleas_node = NODE_CLASS_MAPPINGS["ImageMaskScaleAs"]()
focuscroprestore_node = NODE_CLASS_MAPPINGS["FocusCropRestore"]()
layermask_maskgrow_node = NODE_CLASS_MAPPINGS["LayerMask: MaskGrow"]()


class flux2_klein_faceswap:
    def __init__(self):
        # 预加载模型，风格对齐 flux2_klein_change_bg
        self.cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]().load_clip(
            clip_name="split_files/text_encoders/qwen_3_8b_fp8mixed.safetensors",
            type="flux2",
            device="default",
        )
        self.vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae(vae_name="flux2-vae.safetensors")
        self.unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet(
            unet_name="flux-2-klein-9b-fp8.safetensors", weight_dtype="default"
        )
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )
        NODE_CLASS_MAPPINGS["FaceAnalysisModels"]().load_models(
            library="insightface", provider="CUDA"
        )

    @torch.inference_mode()
    def forward(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """
        图像2人物换上图像1的头和脸，保持图像1光影自然。
        image1: 源人物图，image2: 目标脸图。
        返回换脸后的 PIL.Image。
        """
        cr_text_20 = cr_text_node.text_multiline(
            text="参照图像1和图像2，图像1人物换上图像2的头和脸，保持图像1光影自然"
        )

        loadimage_94 = self.loadimage.load_image(image=image2)
        loadimage_95 = self.loadimage.load_image(image=image1)

        facedetector_87 = facedetector_node.call(
            fit="all",
            expand_rate=0.5,
            only_one=True,
            invert=False,
            input_image=get_value_at_index(loadimage_94, 0),
        )
        focuscropultra_126 = focuscropultra_node.crop_by_mask_v2(
            up_keep=0.3,
            down_keep=0.3,
            right_keep=0.3,
            left_keep=0.3,
            crop_ratio="original",
            aspect_ratio="original",
            fit="fill",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1024,
            background_color="#000000",
            expand=0,
            blur_radius=0,
            image=get_value_at_index(loadimage_94, 0),
            mask=get_value_at_index(facedetector_87, 1),
        )
        getimagesize_109 = getimagesize_node.execute(
            image=get_value_at_index(focuscropultra_126, 3)
        )

        facedetector_77 = facedetector_node.call(
            fit="all",
            expand_rate=0.5,
            only_one=True,
            invert=False,
            input_image=get_value_at_index(loadimage_95, 0),
        )
        focuscropultra_90 = focuscropultra_node.crop_by_mask_v2(
            up_keep=0.2,
            down_keep=0.1,
            right_keep=0.1,
            left_keep=0.1,
            crop_ratio="original",
            aspect_ratio="original",
            fit="fill",
            method="lanczos",
            round_to_multiple="8",
            scale_to_side="longest",
            scale_to_length=1024,
            background_color="#000000",
            expand=0,
            blur_radius=0,
            image=get_value_at_index(loadimage_95, 0),
            mask=get_value_at_index(facedetector_77, 1),
        )
        emptyimagepro_78 = emptyimagepro_node.generate(
            batch_size=1,
            color="255,255,255",
            image=get_value_at_index(focuscropultra_90, 3),
        )
        layermask_personmaskultra_79 = layermask_personmaskultra_node.person_mask_ultra(
            face=True,
            hair=False,
            body=False,
            clothes=False,
            accessories=False,
            background=False,
            confidence=0.4,
            detail_range=16,
            black_point=0.01,
            white_point=0.99,
            process_detail=False,
            images=get_value_at_index(focuscropultra_90, 3),
        )
        facesegment_85 = facesegment_node.segment_face(
            Skin=False,
            Nose=False,
            Eyeglasses=True,
            Left_eye=False,
            Right_eye=False,
            Left_eyebrow=False,
            Right_eyebrow=False,
            Left_ear=False,
            Right_ear=False,
            Mouth=False,
            Upper_lip=False,
            Lower_lip=False,
            Hair=False,
            Earring=False,
            Neck=False,
            process_res=512,
            mask_blur=0,
            mask_offset=0,
            invert_output=False,
            background="Alpha",
            background_color="#222222",
            images=get_value_at_index(focuscropultra_90, 3),
        )
        maskcomposite_89 = maskcomposite_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(layermask_personmaskultra_79, 1),
            source=get_value_at_index(facesegment_85, 1),
        )
        growmask_74 = growmask_node.EXECUTE_NORMALIZED(
            expand=10,
            tapered_corners=False,
            mask=get_value_at_index(maskcomposite_89, 0),
        )
        imagecompositemasked_92 = imagecompositemasked_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(emptyimagepro_78, 0),
            source=get_value_at_index(focuscropultra_90, 3),
            mask=get_value_at_index(growmask_74, 0),
        )
        layermask_personmaskultra_127 = layermask_personmaskultra_node.person_mask_ultra(
            face=True,
            hair=False,
            body=False,
            clothes=False,
            accessories=False,
            background=False,
            confidence=0.4,
            detail_range=16,
            black_point=0.01,
            white_point=0.99,
            process_detail=False,
            images=get_value_at_index(focuscropultra_126, 3),
        )
        maskcomposite_128 = maskcomposite_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(layermask_personmaskultra_127, 1),
            source=get_value_at_index(focuscropultra_126, 4),
        )
        layermask_maskgrow_132 = layermask_maskgrow_node.mask_grow(
            invert_mask=False,
            grow=5,
            blur=5,
            mask=get_value_at_index(maskcomposite_128, 0),
        )
        painterfluximageedit_35 = painterfluximageedit_node.encode(
            prompt=get_value_at_index(cr_text_20, 0),
            width=get_value_at_index(getimagesize_109, 0),
            height=get_value_at_index(getimagesize_109, 1),
            clip=get_value_at_index(self.cliploader, 0),
            vae=get_value_at_index(self.vaeloader, 0),
            image1=get_value_at_index(focuscropultra_126, 3),
            image2=get_value_at_index(imagecompositemasked_92, 0),
            image1_mask=get_value_at_index(layermask_maskgrow_132, 0),
        )
        lanpaint_ksampler_138 = lanpaint_ksampler_node.sample(
            seed=random.randint(1, 2**64),
            steps=4,
            cfg=1,
            sampler_name="euler",
            scheduler="simple",
            denoise=1,
            LanPaint_NumSteps=1,
            LanPaint_PromptMode="Image First",
            LanPaint_Info="LanPaint KSampler. For more info, visit https://github.com/scraed/LanPaint. If you find it useful, please give a star ⭐️!",
            Inpainting_mode="🖼️ Image Inpainting",
            model=get_value_at_index(self.unetloader, 0),
            positive=get_value_at_index(painterfluximageedit_35, 0),
            negative=get_value_at_index(painterfluximageedit_35, 1),
            latent_image=get_value_at_index(painterfluximageedit_35, 2),
        )
        vaedecode_37 = vaedecode_node.decode(
            samples=get_value_at_index(lanpaint_ksampler_138, 0),
            vae=get_value_at_index(self.vaeloader, 0),
        )
        colormatch_103 = colormatch_node.colormatch(
            method="mkl",
            strength=0.75,
            multithread=True,
            image_ref=get_value_at_index(focuscropultra_126, 3),
            image_target=get_value_at_index(vaedecode_37, 0),
        )
        imagemaskscaleas_133 = imagemaskscaleas_node.image_mask_scale_as(
            fit="letterbox",
            method="lanczos",
            scale_as=get_value_at_index(focuscropultra_126, 0),
            mask=get_value_at_index(maskcomposite_128, 0),
        )
        focuscroprestore_107 = focuscroprestore_node.restore_crop_box(
            invert_mask=False,
            expand=10,
            blur_radius=10,
            fill_holes=False,
            background_image=get_value_at_index(loadimage_94, 0),
            croped_image=get_value_at_index(colormatch_103, 0),
            crop_box=get_value_at_index(focuscropultra_126, 2),
            original_size=get_value_at_index(focuscropultra_126, 5),
            croped_mask=get_value_at_index(imagemaskscaleas_133, 1),
        )
        return tensor2pil(get_value_at_index(focuscroprestore_107, 0))



