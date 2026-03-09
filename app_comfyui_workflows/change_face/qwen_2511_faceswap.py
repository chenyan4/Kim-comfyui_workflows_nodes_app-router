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

# 节点实例（与 qwen_change_bg 风格一致，模块级缓存）
unetloader_node = NODE_CLASS_MAPPINGS["UNETLoader"]()
loraloadermodelonly_node = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
cliploader_node = NODE_CLASS_MAPPINGS["CLIPLoader"]()
vaeloader_node = NODE_CLASS_MAPPINGS["VAELoader"]()
unetloadergguf_node = NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]()
modelsamplingauraflow_node = NODE_CLASS_MAPPINGS["ModelSamplingAuraFlow"]()
cfgnorm_node = NODE_CLASS_MAPPINGS["CFGNorm"]()
text_multiline_node = NODE_CLASS_MAPPINGS["Text Multiline"]()
focuscropultra_node = NODE_CLASS_MAPPINGS["FocusCropUltra"]()
vaeencode_node = NODE_CLASS_MAPPINGS["VAEEncode"]()
emptyimagepro_node = NODE_CLASS_MAPPINGS["EmptyImagePro"]()
layermask_personmaskultra_node = NODE_CLASS_MAPPINGS["LayerMask: PersonMaskUltra"]()
facesegment_node = NODE_CLASS_MAPPINGS["FaceSegment"]()
maskcomposite_node = NODE_CLASS_MAPPINGS["MaskComposite"]()
growmask_node = NODE_CLASS_MAPPINGS["GrowMask"]()
imagecompositemasked_node = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
textencodeqwenimageeditplus_node = NODE_CLASS_MAPPINGS["TextEncodeQwenImageEditPlus"]()
conditioningzeroout_node = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
imagemaskscaleas_node = NODE_CLASS_MAPPINGS["ImageMaskScaleAs"]()
layermask_maskgrow_node = NODE_CLASS_MAPPINGS["LayerMask: MaskGrow"]()
cropbymaskv2_node = NODE_CLASS_MAPPINGS["CropByMaskV2"]()
dwpreprocessor_node = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
lanpaint_ksampler_node = NODE_CLASS_MAPPINGS["LanPaint_KSampler"]()
vaedecode_node = NODE_CLASS_MAPPINGS["VAEDecode"]()
colormatch_node = NODE_CLASS_MAPPINGS["ColorMatch"]()
focuscroprestore_node = NODE_CLASS_MAPPINGS["FocusCropRestore"]()
setlatentnoisemask_node = NODE_CLASS_MAPPINGS["SetLatentNoiseMask"]()
layerutility_cropboxresolve_node = NODE_CLASS_MAPPINGS["LayerUtility: CropBoxResolve"]()
simplemath_node = NODE_CLASS_MAPPINGS["SimpleMath+"]()
jwimageresize_node = NODE_CLASS_MAPPINGS["JWImageResize"]()
yolov8_detect = NODE_CLASS_MAPPINGS["yolov8_detect"]()


class qwen_2511_faceswap:
    def __init__(self):
        # 预加载模型，风格对齐 qwen_change_bg
        self.unetloader_43 = unetloader_node.load_unet(
            unet_name="qwen/qwen_image_edit_2511_bf16.safetensors",
            weight_dtype="default",
        )
        self.loraloadermodelonly_1 = loraloadermodelonly_node.load_lora_model_only(
            lora_name="qwen/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-fp32.safetensors",
            strength_model=1,
            model=get_value_at_index(self.unetloader_43, 0),
        )
        self.cliploader_4 = cliploader_node.load_clip(
            clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            type="qwen_image",
            device="default",
        )
        self.vaeloader_6 = vaeloader_node.load_vae(vae_name="qwen_image_vae.safetensors")
        
        self.loraloadermodelonly_151 = loraloadermodelonly_node.load_lora_model_only(
            lora_name="qwen/qwen_F2P_lora.safetensors",
            strength_model=0.7,
            model=get_value_at_index(self.loraloadermodelonly_1, 0),
        )

        # 支持 PIL.Image 的 LoadImage 节点
        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self, image1: Image.Image, image2: Image.Image) -> Image.Image:
        """
        图像2人物换上图像1的头和脸。image1: 源人物图，image2: 目标脸图。
        返回换脸后的 PIL.Image。
        """
        # image1 目标人物，image2 源脸
        loadimage_68 = self.loadimage.load_image(image=image1)
        loadimage_259 = self.loadimage.load_image(image=image2)

        text_multiline_41 = text_multiline_node.text_multiline(
            text="面部替换:把第一张图片中人物的面部换成第二张图片中人物的面部特征"
        )


        yolov8_detect_152 = yolov8_detect.yolo_detect(
            yolo_model="face_yolov8m-seg_60.pt",
            mask_merge="all",
            conf_threshold=0.25,
            image=get_value_at_index(loadimage_259, 0),
        )

        focuscropultra_252 = focuscropultra_node.crop_by_mask_v2(
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
            image=get_value_at_index(loadimage_259, 0),
            mask=get_value_at_index(yolov8_detect_152, 0),
        )
        vaeencode_91 = vaeencode_node.encode(
            pixels=get_value_at_index(focuscropultra_252, 3),
            vae=get_value_at_index(self.vaeloader_6, 0),
        )

        modelsamplingauraflow_2 = modelsamplingauraflow_node.patch_aura(
            shift=3, model=get_value_at_index(self.loraloadermodelonly_151, 0)
        )
        cfgnorm_3 = cfgnorm_node.EXECUTE_NORMALIZED(
            strength=1, model=get_value_at_index(modelsamplingauraflow_2, 0)
        )


        yolov8_detect_153 = yolov8_detect.yolo_detect(
            yolo_model="face_yolov8m-seg_60.pt",
            mask_merge="all",
            conf_threshold=0.25,
            image=get_value_at_index(loadimage_68, 0),
        )

        focuscropultra_137 = focuscropultra_node.crop_by_mask_v2(
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
            image=get_value_at_index(loadimage_68, 0),
            mask=get_value_at_index(yolov8_detect_153, 0),
        )
        emptyimagepro_143 = emptyimagepro_node.generate(
            batch_size=1,
            color="255,255,255",
            image=get_value_at_index(focuscropultra_137, 3),
        )
        layermask_personmaskultra_139 = layermask_personmaskultra_node.person_mask_ultra(
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
            images=get_value_at_index(focuscropultra_137, 3),
        )
        facesegment_141 = facesegment_node.segment_face(
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
            images=get_value_at_index(focuscropultra_137, 3),
        )
        maskcomposite_142 = maskcomposite_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(layermask_personmaskultra_139, 1),
            source=get_value_at_index(facesegment_141, 1),
        )
        growmask_144 = growmask_node.EXECUTE_NORMALIZED(
            expand=10,
            tapered_corners=False,
            mask=get_value_at_index(maskcomposite_142, 0),
        )
        imagecompositemasked_145 = imagecompositemasked_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            resize_source=False,
            destination=get_value_at_index(emptyimagepro_143, 0),
            source=get_value_at_index(focuscropultra_137, 3),
            mask=get_value_at_index(growmask_144, 0),
        )
        textencodeqwenimageeditplus_39 = (
            textencodeqwenimageeditplus_node.EXECUTE_NORMALIZED(
                prompt=get_value_at_index(text_multiline_41, 0),
                clip=get_value_at_index(self.cliploader_4, 0),
                vae=get_value_at_index(self.vaeloader_6, 0),
                image1=get_value_at_index(focuscropultra_252, 3),
                image2=get_value_at_index(imagecompositemasked_145, 0),
            )
        )
        conditioningzeroout_12 = conditioningzeroout_node.zero_out(
            conditioning=get_value_at_index(textencodeqwenimageeditplus_39, 0)
        )
        imagemaskscaleas_15 = imagemaskscaleas_node.image_mask_scale_as(
            fit="letterbox",
            method="lanczos",
            scale_as=get_value_at_index(focuscropultra_252, 3),
            image=get_value_at_index(imagecompositemasked_145, 0),
        )
        layermask_personmaskultra_249 = layermask_personmaskultra_node.person_mask_ultra(
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
            images=get_value_at_index(focuscropultra_252, 3),
        )
        maskcomposite_255 = maskcomposite_node.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(layermask_personmaskultra_249, 1),
            source=get_value_at_index(focuscropultra_252, 4),
        )
        layermask_maskgrow_257 = layermask_maskgrow_node.mask_grow(
            invert_mask=False,
            grow=5,
            blur=5,
            mask=get_value_at_index(maskcomposite_255, 0),
        )
        cropbymaskv2_24 = cropbymaskv2_node.crop_by_mask_v2(
            invert_mask=False,
            detect="mask_area",
            top_reserve=20,
            bottom_reserve=20,
            left_reserve=20,
            right_reserve=20,
            round_to_multiple="8",
            image=get_value_at_index(loadimage_259, 0),
            mask=get_value_at_index(layermask_maskgrow_257, 0),
            crop_box=get_value_at_index(focuscropultra_252, 2),
        )
        dwpreprocessor_25 = dwpreprocessor_node.estimate_pose(
            detect_hand="enable",
            detect_body="enable",
            detect_face="enable",
            resolution=512,
            bbox_detector="yolox_l.onnx",
            pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
            scale_stick_for_xinsr_cn="disable",
            image=get_value_at_index(cropbymaskv2_24, 0),
        )
        lanpaint_ksampler_95 = lanpaint_ksampler_node.sample(
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
            model=get_value_at_index(cfgnorm_3, 0),
            positive=get_value_at_index(textencodeqwenimageeditplus_39, 0),
            negative=get_value_at_index(conditioningzeroout_12, 0),
            latent_image=get_value_at_index(vaeencode_91, 0),
        )
        vaedecode_31 = vaedecode_node.decode(
            samples=get_value_at_index(lanpaint_ksampler_95, 0),
            vae=get_value_at_index(self.vaeloader_6, 0),
        )
        colormatch_127 = colormatch_node.colormatch(
            method="mkl",
            strength=1,
            multithread=True,
            image_ref=get_value_at_index(focuscropultra_252, 3),
            image_target=get_value_at_index(vaedecode_31, 0),
        )
        imagemaskscaleas_247 = imagemaskscaleas_node.image_mask_scale_as(
            fit="letterbox",
            method="lanczos",
            scale_as=get_value_at_index(focuscropultra_252, 0),
            mask=get_value_at_index(maskcomposite_255, 0),
        )
        layermask_maskgrow_116 = layermask_maskgrow_node.mask_grow(
            invert_mask=False,
            grow=10,
            blur=10,
            mask=get_value_at_index(imagemaskscaleas_247, 1),
        )
        focuscroprestore_30 = focuscroprestore_node.restore_crop_box(
            invert_mask=False,
            expand=10,
            blur_radius=8,
            fill_holes=False,
            background_image=get_value_at_index(loadimage_259, 0),
            croped_image=get_value_at_index(colormatch_127, 0),
            crop_box=get_value_at_index(focuscropultra_252, 2),
            original_size=get_value_at_index(focuscropultra_252, 5),
            croped_mask=get_value_at_index(layermask_maskgrow_116, 0),
        )
    
        return tensor2pil(get_value_at_index(focuscroprestore_30, 0))


# if __name__ == "__main__":
#     processor = qwen_2511_faceswap()
#     image1 = Image.open("../my_images/15-0.jpeg")
#     image2 = Image.open("../my_images/15-2.jpeg")
#     res = processor.forward(image1, image2)
#     res.save("../my_images/faceswap_res_qwen.jpg")
