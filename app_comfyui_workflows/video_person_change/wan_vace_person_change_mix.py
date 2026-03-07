import os
import random
import sys
import types
from functools import wraps
from typing import Sequence, Mapping, Any, Union
import numpy as np
import torch


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

# 节点实例（与 wan_vace_person_change_one 风格一致，模块级缓存）
intconstant_node = NODE_CLASS_MAPPINGS["INTConstant"]()
primitivefloat_node = NODE_CLASS_MAPPINGS["PrimitiveFloat"]()
loadimage_node = NODE_CLASS_MAPPINGS["LoadImage"]()
wanvideovaeloader_node = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
wanvideoblockswap_node = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
wanvideotextencodecached_node = NODE_CLASS_MAPPINGS["WanVideoTextEncodeCached"]()
clipvisionloader_node = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
sdposeoodloader_node = NODE_CLASS_MAPPINGS["SDPoseOODLoader"]()
wanvideoloraselectmulti_node = NODE_CLASS_MAPPINGS["WanVideoLoraSelectMulti"]()
wanvideomodelloader_node = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
yolomodelloader_node = NODE_CLASS_MAPPINGS["YOLOModelLoader"]()
wanvideosetloras_node = NODE_CLASS_MAPPINGS["WanVideoSetLoRAs"]()
wanvideosetblockswap_node = NODE_CLASS_MAPPINGS["WanVideoSetBlockSwap"]()
vhs_loadvideo_node = NODE_CLASS_MAPPINGS["VHS_LoadVideo"]()
imageresizekjv2_node = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
wanvideoclipvisionencode_node = NODE_CLASS_MAPPINGS["WanVideoClipVisionEncode"]()
sdposeoodprocessor_node = NODE_CLASS_MAPPINGS["SDPoseOODProcessor"]()
wanvideoanimateembeds_node = NODE_CLASS_MAPPINGS["WanVideoAnimateEmbeds"]()
wanvideosampler_node = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
wanvideodecode_node = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
vhs_videocombine_node = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

# 默认负向提示
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class WanVacePersonChangeMix:
    def __init__(self):
        # 预加载模型，风格对齐 wan_vace_person_change_one
        self.vae = wanvideovaeloader_node.loadmodel(
            model_name="wan_2.1_vae.safetensors", precision="bf16", use_cpu_cache=False
        )
        self.blockswap_args = wanvideoblockswap_node.setargs(
            blocks_to_swap=40,
            offload_img_emb=False,
            offload_txt_emb=False,
            use_non_blocking=True,
            vace_blocks_to_swap=0,
            prefetch_blocks=0,
            block_swap_debug=False,
        )
        self.textencodecached = wanvideotextencodecached_node.process(
            model_name="text_encoders/umt5-xxl-enc-fp8_e4m3fn.safetensors",
            precision="bf16",
            positive_prompt="",
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            quantization="disabled",
            use_disk_cache=False,
            device="gpu",
        )
        self.clip_vision = clipvisionloader_node.load_clip(
            clip_name="clip_vision_h.safetensors"
        )
        self.sdpose_model = sdposeoodloader_node.load_sdpose_model(
            model_type="WholeBody",
            unet_precision="fp16",
            device="auto",
            unload_on_finish=True,
        )
        self.lora = wanvideoloraselectmulti_node.getlorapath(
            lora_0="WanAnimate_relight_lora_fp16.safetensors",
            strength_0=1,
            lora_1="Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
            strength_1=1,
            lora_2="FastWan/FastWan_T2V_14B_480p_lora_rank_128_bf16.safetensors",
            strength_2=1,
            lora_3="Pusa/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors",
            strength_3=1,
            lora_4="Wan2.2-Fun-A14B-InP-LOW-HPS2.1_resized_dynamic_avg_rank_15_bf16.safetensors",
            strength_4=0.5,
            low_mem_load=False,
            merge_loras=False,
        )
        self.yolo_model = yolomodelloader_node.load_yolo(model_name="yolo11x.pt")
        model_raw = wanvideomodelloader_node.loadmodel(
            model="Wan22Animate/Wan2_2-Animate-14B_fp8_scaled_e4m3fn_KJ_v2.safetensors",
            base_precision="fp16",
            quantization="fp8_e4m3fn_scaled",
            load_device="offload_device",
            attention_mode="sageattn",
            rms_norm_function="default",
        )
        model_with_lora = wanvideosetloras_node.setlora(
            model=get_value_at_index(model_raw, 0),
            lora=get_value_at_index(self.lora, 0),
        )
        self.model = wanvideosetblockswap_node.loadmodel(
            model=get_value_at_index(model_with_lora, 0),
            block_swap_args=get_value_at_index(self.blockswap_args, 0),
        )

        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self,image_1=None,video_url=None,width=840,height=1024,fps=16):
        """
        视频换人（Mix 版）：参考人物图 + 原视频，用 YOLO+SDPose 检测 pose/face -> AnimateEmbeds -> 采样 -> 解码 -> 合成，返回本地视频路径。
        """
        width_const = intconstant_node.get_value(value=width)
        height_const = intconstant_node.get_value(value=height)
        fps_val = primitivefloat_node.EXECUTE_NORMALIZED(value=fps)

        load_ref = self.loadimage.load_image(image=image_1)
        vhs_loadvideo_out = vhs_loadvideo_node.load_video(
            video=video_url,
            force_rate=int(fps),
            custom_width=get_value_at_index(width_const, 0),
            custom_height=get_value_at_index(height_const, 0),
            frame_load_cap=0,
            skip_first_frames=0,
            select_every_nth=1,
            format="AnimateDiff",
            unique_id=16943214551238001183,
        )
        ref_resized = imageresizekjv2_node.resize(
            width=get_value_at_index(width_const, 0),
            height=get_value_at_index(height_const, 0),
            upscale_method="lanczos",
            keep_proportion="crop",
            pad_color="0, 0, 0",
            crop_position="center",
            divisible_by=16,
            device="cpu",
            image=get_value_at_index(load_ref, 0),
            unique_id=5979623874891116213,
        )
        clip_embeds = wanvideoclipvisionencode_node.process(
            strength_1=1,
            strength_2=1,
            crop="center",
            combine_embeds="average",
            force_offload=True,
            tiles=0,
            ratio=0.5,
            clip_vision=get_value_at_index(self.clip_vision, 0),
            image_1=get_value_at_index(ref_resized, 0),
        )
        sdpose_out = sdposeoodprocessor_node.process_sequence(
            score_threshold=0.3,
            overlay_alpha=1,
            batch_size=1,
            prompt="person .",
            gd_threshold=0.3,
            save_for_editor=False,
            filename_prefix_edit="poses/pose_edit",
            keep_face=True,
            keep_hands=True,
            keep_feet=True,
            scale_for_xinsr=True,
            pose_scale_factor=1,
            sdpose_model=get_value_at_index(self.sdpose_model, 0),
            images=get_value_at_index(vhs_loadvideo_out, 0),
            yolo_model=get_value_at_index(self.yolo_model, 0),
        )
        animate_embeds = wanvideoanimateembeds_node.process(
            width=get_value_at_index(width_const, 0),
            height=get_value_at_index(height_const, 0),
            num_frames=get_value_at_index(vhs_loadvideo_out, 1),
            force_offload=True,
            frame_window_size=81,
            colormatch="disabled",
            pose_strength=1,
            face_strength=1,
            tiled_vae=False,
            vae=get_value_at_index(self.vae, 0),
            clip_embeds=get_value_at_index(clip_embeds, 0),
            ref_images=get_value_at_index(ref_resized, 0),
            pose_images=get_value_at_index(sdpose_out, 0),
            face_images=get_value_at_index(sdpose_out, 0),
        )
        sampler_out = wanvideosampler_node.process(
            steps=4,
            cfg=1,
            shift=5.000000000000001,
            seed=random.randint(1, 2**64),
            force_offload=True,
            scheduler="dpm++_sde",
            riflex_freq_index=0,
            denoise_strength=1,
            batched_cfg="",
            rope_function="comfy",
            start_step=0,
            end_step=-1,
            add_noise_to_samples=False,
            model=get_value_at_index(self.model, 0),
            image_embeds=get_value_at_index(animate_embeds, 0),
            text_embeds=get_value_at_index(self.textencodecached, 0),
        )
        decode_out = wanvideodecode_node.decode(
            enable_vae_tiling=False,
            tile_x=272,
            tile_y=272,
            tile_stride_x=144,
            tile_stride_y=128,
            normalization="default",
            vae=get_value_at_index(self.vae, 0),
            samples=get_value_at_index(sampler_out, 0),
        )
        vhs_videocombine_out = vhs_videocombine_node.combine_video(
            frame_rate=get_value_at_index(fps_val, 0),
            loop_count=0,
            filename_prefix="WanAnimate_person_change_mix",
            format="video/h264-mp4",
            pix_fmt="yuv420p",
            crf=19,
            save_metadata=True,
            trim_to_audio=True,
            pingpong=False,
            save_output=True,
            images=get_value_at_index(decode_out, 0),
            audio=get_value_at_index(vhs_loadvideo_out, 2),
            unique_id=1068530801337745208,
        )
        video = get_value_at_index(vhs_videocombine_out, 0)[1][-1]
        return video


