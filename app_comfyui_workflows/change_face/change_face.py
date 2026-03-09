import os
import random
import sys
import gc
from typing import Sequence, Mapping, Any, Union
import torch
import numpy as np
from PIL import Image
import types
from functools import wraps


def support_pil_image(original_method):
    @wraps(original_method)
    def wrapper(self, image=None, *args, **kwargs):
        if hasattr(image, 'save'):
            channel = kwargs.get("channel", None)
            image = image.convert('RGB' if channel is None else "L")
            return (pil2tensor(image), )
        return original_method(self, image=image, *args, **kwargs)
    return wrapper


def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    new_tensor = torch.tensor(img_array)
    return new_tensor.unsqueeze(0)


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
    comfyui_path = '/data/chenyan/comfyui'
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

yolov8_detect = NODE_CLASS_MAPPINGS["yolov8_detect"]()
instantidmodelloader = NODE_CLASS_MAPPINGS["InstantIDModelLoader"]()
instantidfaceanalysis = NODE_CLASS_MAPPINGS["InstantIDFaceAnalysis"]()
checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
diffcontrolnetloader = NODE_CLASS_MAPPINGS["DiffControlNetLoader"]()
loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
focuscropultra = NODE_CLASS_MAPPINGS["FocusCropUltra"]()
emptyimagepro = NODE_CLASS_MAPPINGS["EmptyImagePro"]()
layermask_personmaskultra = NODE_CLASS_MAPPINGS["LayerMask: PersonMaskUltra"]()
facesegment = NODE_CLASS_MAPPINGS["FaceSegment"]()
maskcomposite = NODE_CLASS_MAPPINGS["MaskComposite"]()
imagecomposite = NODE_CLASS_MAPPINGS["ImageComposite+"]()
cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
cr_text = NODE_CLASS_MAPPINGS["CR Text"]()
applyinstantid = NODE_CLASS_MAPPINGS["ApplyInstantID"]()
growmask = NODE_CLASS_MAPPINGS["GrowMask"]()
maskblur = NODE_CLASS_MAPPINGS["MaskBlur+"]()
inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
pulidmodelloader = NODE_CLASS_MAPPINGS["PulidModelLoader"]()
pulidevacliploader = NODE_CLASS_MAPPINGS["PulidEvaClipLoader"]()
pulidinsightfaceloader = NODE_CLASS_MAPPINGS["PulidInsightFaceLoader"]()
applypulid = NODE_CLASS_MAPPINGS["ApplyPulid"]()
rescalecfg = NODE_CLASS_MAPPINGS["RescaleCFG"]()
ipadapterunifiedloaderfaceid = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoaderFaceID"]()
ipadapterfaceid = NODE_CLASS_MAPPINGS["IPAdapterFaceID"]()
ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
colormatch = NODE_CLASS_MAPPINGS["ColorMatch"]()
imagetomask = NODE_CLASS_MAPPINGS["ImageToMask"]()
easy_imagesize = NODE_CLASS_MAPPINGS["easy imageSize"]()
imageresize = NODE_CLASS_MAPPINGS["ImageResize+"]()
imagecompositemasked = NODE_CLASS_MAPPINGS["ImageCompositeMasked"]()
constrainimagepysssss = NODE_CLASS_MAPPINGS["ConstrainImage|pysssss"]()
masktoimage = NODE_CLASS_MAPPINGS["MaskToImage"]()
imagemaskscaleas = NODE_CLASS_MAPPINGS["ImageMaskScaleAs"]()
layerutility_cropboxresolve = NODE_CLASS_MAPPINGS["LayerUtility: CropBoxResolve"]()
saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()
maskpreview = NODE_CLASS_MAPPINGS["MaskPreview+"]()


class ChangeFace:
    def __init__(self):
        self.instantidmodelloader_384 = instantidmodelloader.load_model(
            instantid_file="ip-adapter.bin"
        )

        self.instantidfaceanalysis_385 = instantidfaceanalysis.load_insight_face(
            provider="CUDA"
        )

        self.checkpointloadersimple_504 = checkpointloadersimple.load_checkpoint(
            ckpt_name="真境写真XL Apex _ 商业电商摄影真实写实大师_真境写真XL_v6.safetensors"
        )

        self.diffcontrolnetloader_386 = diffcontrolnetloader.load_controlnet(
            control_net_name="control_instant_id_sdxl.safetensors",
            model=get_value_at_index(self.checkpointloadersimple_504, 0),
        )

        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage
        )

    @torch.inference_mode()
    def forward(self, init_image, userdefined_image):
        loadimage_502 = self.loadimage.load_image(image=init_image)
        loadimage_503 = self.loadimage.load_image(image=userdefined_image)

        yolov8_detect_531 = yolov8_detect.yolo_detect(
            yolo_model="face_yolov8n-seg2_60.pt",
            mask_merge="all",
            conf_threshold=0.25,
            image=get_value_at_index(loadimage_502, 0),
        )

        focuscropultra_505 = focuscropultra.crop_by_mask_v2(
            up_keep=0.2,
            down_keep=0.1,
            right_keep=0.1,
            left_keep=0.1,
            crop_ratio="original",
            aspect_ratio="original",
            fit="fill",
            method="lanczos",
            round_to_multiple=8,
            scale_to_side="longest",
            scale_to_length=1024,
            background_color="#000000",
            expand=0,
            blur_radius=0,
            image=get_value_at_index(loadimage_502, 0),
            mask=get_value_at_index(yolov8_detect_531, 0),
        )

        emptyimagepro_489 = emptyimagepro.generate(
            batch_size=1,
            color="255,255,255",
            image=get_value_at_index(focuscropultra_505, 3),
        )

        layermask_personmaskultra_506 = layermask_personmaskultra.person_mask_ultra(
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
            images=get_value_at_index(focuscropultra_505, 3),
        )

        facesegment_507 = facesegment.segment_face(
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
            images=get_value_at_index(focuscropultra_505, 3),
        )

        maskcomposite_491 = maskcomposite.EXECUTE_NORMALIZED(
            x=0,
            y=0,
            operation="or",
            destination=get_value_at_index(layermask_personmaskultra_506, 1),
            source=get_value_at_index(facesegment_507, 1),
        )

        growmask_492 = growmask.EXECUTE_NORMALIZED(
            expand=10,
            tapered_corners=True,
            mask=get_value_at_index(maskcomposite_491, 0),
        )

        imagecomposite_496 = imagecomposite.execute(
            x=0,
            y=0,
            offset_x=0,
            offset_y=0,
            destination=get_value_at_index(emptyimagepro_489, 0),
            source=get_value_at_index(focuscropultra_505, 3),
            mask=get_value_at_index(growmask_492, 0),
        )

        cliptextencode_412 = cliptextencode.encode(
            text="clear face,Best quality, 8k, photographic style, real texture",
            clip=get_value_at_index(self.checkpointloadersimple_504, 1),
        )

        cr_text_498 = cr_text.text_multiline(
            text="clear face,Best quality, 8k, photographic style, real texture"
        )

        cliptextencode_408 = cliptextencode.encode(
            text=get_value_at_index(cr_text_498, 0),
            clip=get_value_at_index(self.checkpointloadersimple_504, 1),
        )

        yolov8_detect_535 = yolov8_detect.yolo_detect(
            yolo_model="face_yolov8n-seg2_60.pt",
            mask_merge="all",
            conf_threshold=0.25,
            image=get_value_at_index(loadimage_503, 0),
        )

        focuscropultra_462 = focuscropultra.crop_by_mask_v2(
            up_keep=0.2,
            down_keep=0.1,
            right_keep=0.1,
            left_keep=0.1,
            crop_ratio="original",
            aspect_ratio="original",
            fit="fill",
            method="lanczos",
            round_to_multiple=8,
            scale_to_side="longest",
            scale_to_length=1024,
            background_color="#000000",
            expand=0,
            blur_radius=0,
            image=get_value_at_index(loadimage_503, 0),
            mask=get_value_at_index(yolov8_detect_535, 0),
        )

        crop_box = get_value_at_index(focuscropultra_462, 2)
        if crop_box is None:
            raise ValueError("目标图片中未检测到人脸，请上传包含清晰人脸的图片")

        colormatch_463 = colormatch.colormatch(
            method="mkl",
            strength=0.7,
            multithread=True,
            image_ref=get_value_at_index(loadimage_503, 0),
            image_target=get_value_at_index(focuscropultra_462, 3),
        )

        applyinstantid_395 = applyinstantid.apply_instantid(
            weight=0.9,
            start_at=0,
            end_at=1,
            instantid=get_value_at_index(self.instantidmodelloader_384, 0),
            insightface=get_value_at_index(self.instantidfaceanalysis_385, 0),
            control_net=get_value_at_index(self.diffcontrolnetloader_386, 0),
            image=get_value_at_index(imagecomposite_496, 0),
            model=get_value_at_index(self.checkpointloadersimple_504, 0),
            positive=get_value_at_index(cliptextencode_412, 0),
            negative=get_value_at_index(cliptextencode_408, 0),
            image_kps=get_value_at_index(colormatch_463, 0),
        )

        layermask_personmaskultra_510 = layermask_personmaskultra.person_mask_ultra(
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
            images=get_value_at_index(focuscropultra_462, 0),
        )

        imagemaskscaleas_516 = imagemaskscaleas.image_mask_scale_as(
            fit="letterbox",
            method="lanczos",
            scale_as=get_value_at_index(focuscropultra_462, 3),
            mask=get_value_at_index(layermask_personmaskultra_510, 1),
        )

        growmask_523 = growmask.EXECUTE_NORMALIZED(
            expand=30,
            tapered_corners=False,
            mask=get_value_at_index(imagemaskscaleas_516, 1),
        )

        maskblur_522 = maskblur.execute(
            amount=30, device="auto", mask=get_value_at_index(growmask_523, 0)
        )

        inpaintmodelconditioning_393 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(applyinstantid_395, 1),
            negative=get_value_at_index(applyinstantid_395, 2),
            vae=get_value_at_index(self.checkpointloadersimple_504, 2),
            pixels=get_value_at_index(colormatch_463, 0),
            mask=get_value_at_index(maskblur_522, 0),
        )

        pulidmodelloader_416 = pulidmodelloader.load_model(
            pulid_file="ip-adapter_pulid_sdxl_fp16.safetensors"
        )

        pulidevacliploader_417 = pulidevacliploader.load_eva_clip()

        pulidinsightfaceloader_418 = pulidinsightfaceloader.load_insightface(
            provider="CUDA"
        )

        applypulid_396 = applypulid.apply_pulid(
            method="fidelity",
            weight=1,
            start_at=0,
            end_at=1,
            model=get_value_at_index(applyinstantid_395, 0),
            pulid=get_value_at_index(pulidmodelloader_416, 0),
            eva_clip=get_value_at_index(pulidevacliploader_417, 0),
            face_analysis=get_value_at_index(pulidinsightfaceloader_418, 0),
            image=get_value_at_index(imagecomposite_496, 0),
        )

        rescalecfg_397 = rescalecfg.patch(
            multiplier=0.7, model=get_value_at_index(applypulid_396, 0)
        )

        ipadapterunifiedloaderfaceid_446 = ipadapterunifiedloaderfaceid.load_models(
            preset="FACEID PLUS V2",
            lora_strength=0.9,
            provider="CUDA",
            model=get_value_at_index(rescalecfg_397, 0),
        )

        ipadapterfaceid_451 = ipadapterfaceid.apply_ipadapter(
            weight=0.9,
            weight_faceidv2=1,
            weight_type="linear",
            combine_embeds="concat",
            start_at=0,
            end_at=1,
            embeds_scaling="V only",
            model=get_value_at_index(self.checkpointloadersimple_504, 0),
            ipadapter=get_value_at_index(ipadapterunifiedloaderfaceid_446, 1),
            image=get_value_at_index(colormatch_463, 0),
        )

        ksampler_414 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=20,
            cfg=2,
            sampler_name="uni_pc_bh2",
            scheduler="karras",
            denoise=1,
            model=get_value_at_index(ipadapterfaceid_451, 0),
            positive=get_value_at_index(inpaintmodelconditioning_393, 0),
            negative=get_value_at_index(inpaintmodelconditioning_393, 0),
            latent_image=get_value_at_index(inpaintmodelconditioning_393, 2),
        )

        vaedecode_448 = vaedecode.decode(
            samples=get_value_at_index(ksampler_414, 0),
            vae=get_value_at_index(self.checkpointloadersimple_504, 2),
        )

        inpaintmodelconditioning_443 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(inpaintmodelconditioning_393, 0),
            negative=get_value_at_index(inpaintmodelconditioning_393, 1),
            vae=get_value_at_index(self.checkpointloadersimple_504, 2),
            pixels=get_value_at_index(vaedecode_448, 0),
            mask=get_value_at_index(maskblur_522, 0),
        )

        for q in range(1):
            ksampler_401 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=1.5,
                sampler_name="uni_pc_bh2",
                scheduler="karras",
                denoise=0.5,
                model=get_value_at_index(applyinstantid_395, 0),
                positive=get_value_at_index(inpaintmodelconditioning_443, 0),
                negative=get_value_at_index(inpaintmodelconditioning_443, 1),
                latent_image=get_value_at_index(inpaintmodelconditioning_443, 2),
            )

            easy_imagesize_422 = easy_imagesize.image_width_height(
                image=get_value_at_index(focuscropultra_462, 0)
            )

            vaedecode_449 = vaedecode.decode(
                samples=get_value_at_index(ksampler_401, 0),
                vae=get_value_at_index(self.checkpointloadersimple_504, 2),
            )

            colormatch_423 = colormatch.colormatch(
                method="mkl",
                strength=1,
                multithread=True,
                image_ref=get_value_at_index(focuscropultra_462, 0),
                image_target=get_value_at_index(vaedecode_449, 0),
            )

            imageresize_420 = imageresize.execute(
                width=get_value_at_index(easy_imagesize_422, 0),
                height=get_value_at_index(easy_imagesize_422, 1),
                interpolation="lanczos",
                method="keep proportion",
                condition="always",
                multiple_of=0,
                image=get_value_at_index(colormatch_423, 0),
            )

            layerutility_cropboxresolve_433 = (
                layerutility_cropboxresolve.crop_box_resolve(
                    crop_box=get_value_at_index(focuscropultra_462, 2)
                )
            )

            composite_x = max(0, get_value_at_index(layerutility_cropboxresolve_433, 0))
            composite_y = max(0, get_value_at_index(layerutility_cropboxresolve_433, 1))

            growmask_514 = growmask.EXECUTE_NORMALIZED(
                expand=40,
                tapered_corners=False,
                mask=get_value_at_index(layermask_personmaskultra_510, 1),
            )

            maskblur_515 = maskblur.execute(
                amount=60, device="auto", mask=get_value_at_index(growmask_514, 0)
            )

            imagecompositemasked_478 = imagecompositemasked.EXECUTE_NORMALIZED(
                x=composite_x,
                y=composite_y,
                resize_source=False,
                destination=get_value_at_index(loadimage_503, 0),
                source=get_value_at_index(imageresize_420, 0),
                mask=get_value_at_index(maskblur_515, 0),
            )

        pil_image = tensor2pil(get_value_at_index(imagecompositemasked_478, 0))

        # 尽量释放中间结果所占用的显存/内存
        try:
            del (
                loadimage_502,
                loadimage_503,
                yolov8_detect_531,
                focuscropultra_505,
                emptyimagepro_489,
                layermask_personmaskultra_506,
                facesegment_507,
                maskcomposite_491,
                growmask_492,
                imagecomposite_496,
                cliptextencode_412,
                cr_text_498,
                cliptextencode_408,
                yolov8_detect_535,
                focuscropultra_462,
                crop_box,
                colormatch_463,
                applyinstantid_395,
                layermask_personmaskultra_510,
                imagemaskscaleas_516,
                growmask_523,
                maskblur_522,
                inpaintmodelconditioning_393,
                pulidmodelloader_416,
                pulidevacliploader_417,
                pulidinsightfaceloader_418,
                applypulid_396,
                rescalecfg_397,
                ipadapterunifiedloaderfaceid_446,
                ipadapterfaceid_451,
                ksampler_414,
                vaedecode_448,
                inpaintmodelconditioning_443,
                ksampler_401,
                easy_imagesize_422,
                vaedecode_449,
                colormatch_423,
                imageresize_420,
                layerutility_cropboxresolve_433,
                composite_x,
                composite_y,
                growmask_514,
                maskblur_515,
                imagecompositemasked_478,
            )
        except Exception:
            pass

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return pil_image
