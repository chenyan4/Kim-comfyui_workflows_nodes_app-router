import os
import random
import sys
import types
import gc
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


def cleanup_memory():
    """
    清理一次 Python 对象和 CUDA 显存，并尽量让 ComfyUI 释放未使用的模型。
    在每个大 workflow 结束后调用。
    """
    try:
        import comfy.model_management as model_management  # type: ignore
    except Exception:
        model_management = None

    try:
        gc.collect()
    except Exception:
        pass

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if model_management is not None:
        try:
            model_management.unload_all_models()
        except Exception:
            pass

# 节点实例（与 wan_vace_i2v 风格一致，模块级缓存）
floatconstant_node = NODE_CLASS_MAPPINGS["FloatConstant"]()
intconstant_node = NODE_CLASS_MAPPINGS["INTConstant"]()
int_node = NODE_CLASS_MAPPINGS["Int"]()
loadimage_node = NODE_CLASS_MAPPINGS["LoadImage"]()
getimagesize_node = NODE_CLASS_MAPPINGS["GetImageSize"]()
getimagesizeandcount_node = NODE_CLASS_MAPPINGS["GetImageSizeAndCount"]()
imageresizekjv2_node = NODE_CLASS_MAPPINGS["ImageResizeKJv2"]()
wanvideovacemodelselect_node = NODE_CLASS_MAPPINGS["WanVideoVACEModelSelect"]()
wanvideovaeloader_node = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
loadwanvideot5textencoder_node = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
wanvideoblockswap_node = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
wanvideoloraselectmulti_node = NODE_CLASS_MAPPINGS["WanVideoLoraSelectMulti"]()
stringtofloatlist_node = NODE_CLASS_MAPPINGS["StringToFloatList"]()
wanvideosigmatostep_node = NODE_CLASS_MAPPINGS["WanVideoSigmaToStep"]()
wanvideoenhanceavideo_node = NODE_CLASS_MAPPINGS["WanVideoEnhanceAVideo"]()
wanvideoeasycache_node = NODE_CLASS_MAPPINGS["WanVideoEasyCache"]()
wanvideoslg_node = NODE_CLASS_MAPPINGS["WanVideoSLG"]()
wanvideoexperimentalargs_node = NODE_CLASS_MAPPINGS["WanVideoExperimentalArgs"]()
wanvideomodelloader_high_node = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
wanvideomodelloader_low_node = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
wanvideoscheduler_node = NODE_CLASS_MAPPINGS["WanVideoScheduler"]()
wanvideosetblockswap_node = NODE_CLASS_MAPPINGS["WanVideoSetBlockSwap"]()
wanvideosetloras_node = NODE_CLASS_MAPPINGS["WanVideoSetLoRAs"]()
wanvideovacestarttoendframe_node = NODE_CLASS_MAPPINGS["WanVideoVACEStartToEndFrame"]()
wanvideovaceencode_node = NODE_CLASS_MAPPINGS["WanVideoVACEEncode"]()
wanvideotextencode_node = NODE_CLASS_MAPPINGS["WanVideoTextEncode"]()
wanvideosampler_node = NODE_CLASS_MAPPINGS["WanVideoSampler"]()
wanvideodecode_node = NODE_CLASS_MAPPINGS["WanVideoDecode"]()
vhs_videocombine_node = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

# 默认负向提示
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class WanVaceHFirstEnd:
    def __init__(self):
        # 预加载模型，风格对齐 wan_vace_i2v
        self.vace_high = wanvideovacemodelselect_node.getvacepath(
            vace_model="Wan2.2_VACE/Wan2_2_Fun_VACE_module_A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
        )
        self.vace_low = wanvideovacemodelselect_node.getvacepath(
            vace_model="Wan2.2_VACE/Wan2_2_Fun_VACE_module_A14B_LOW_fp8_e4m3fn_scaled_KJ.safetensors"
        )
        self.vae = wanvideovaeloader_node.loadmodel(
            model_name="wan_2.1_vae.safetensors", precision="bf16", use_cpu_cache=True
        )
        self.t5 = loadwanvideot5textencoder_node.loadmodel(
            model_name="umt5_xxl_fp16.safetensors",
            precision="bf16",
            load_device="offload_device",
            quantization="disabled",
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
        self.easycache_args = wanvideoeasycache_node.setargs(
            easycache_thresh=0.015,
            start_step=2,
            end_step=-1,
            cache_device="offload_device",
        )
        self.stringtofloatlist_63 = stringtofloatlist_node.createlist(string="0,0,3,3")
        self.stringtofloatlist_66 = stringtofloatlist_node.createlist(string="3.5,3,1,1")
        self.sigmatostep = wanvideosigmatostep_node.convert(sigma=0.9)
        self.enhanceavideo_args = wanvideoenhanceavideo_node.setargs(
            weight=2, start_percent=0, end_percent=1
        )
        self.slg_args = wanvideoslg_node.process(
            blocks="10", start_percent=0.1, end_percent=1
        )
        self.experimentalargs_args = wanvideoexperimentalargs_node.process(
            video_attention_split_steps="",
            cfg_zero_star=True,
            use_zero_init=False,
            zero_star_steps=0,
            use_fresca=False,
            fresca_scale_low=1,
            fresca_scale_high=1.25,
            fresca_freq_cutoff=20,
            use_tcfg=False,
            raag_alpha=0,
            bidirectional_sampling=False,
            temporal_score_rescaling=False,
            tsr_k=0.95,
            tsr_sigma=1,
        )
        self.lora_multi_high = wanvideoloraselectmulti_node.getlorapath(
            lora_0="Lightx2v/lightx2v_elite_it2v_animate_face.safetensors",
            strength_0=get_value_at_index(self.stringtofloatlist_63, 0),
            lora_1="Wan22_FunReward/FullDynamic_Ultimate_Fusion_Elite.safetensors",
            strength_1=1,
            lora_2="none",
            strength_2=1,
            lora_3="none",
            strength_3=1,
            lora_4="none",
            strength_4=1,
            low_mem_load=False,
            merge_loras=False,
        )
        self.lora_multi_low = wanvideoloraselectmulti_node.getlorapath(
            lora_0="Lightx2v/lightx2v_elite_it2v_animate_face.safetensors",
            strength_0=1.5,
            lora_1="Wan22_FunReward/FullDynamic_Ultimate_Fusion_Elite.safetensors",
            strength_1=1,
            lora_2="none",
            strength_2=1,
            lora_3="none",
            strength_3=1,
            lora_4="none",
            strength_4=1,
            low_mem_load=False,
            merge_loras=False,
        )
        model_low_raw = wanvideomodelloader_low_node.loadmodel(
            model="Wan2.2/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            base_precision="fp16_fast",
            quantization="disabled",
            load_device="offload_device",
            attention_mode="sageattn",
            rms_norm_function="default",
            extra_model=get_value_at_index(self.vace_low, 0),
        )
        model_low_swap = wanvideosetblockswap_node.loadmodel(
            model=get_value_at_index(model_low_raw, 0),
            block_swap_args=get_value_at_index(self.blockswap_args, 0),
        )
        self.model_low = wanvideosetloras_node.setlora(
            model=get_value_at_index(model_low_swap, 0),
            lora=get_value_at_index(self.lora_multi_low, 0),
        )
        model_high_raw = wanvideomodelloader_high_node.loadmodel(
            model="Wan2.2/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            base_precision="fp16_fast",
            quantization="disabled",
            load_device="offload_device",
            attention_mode="sageattn",
            rms_norm_function="default",
            extra_model=get_value_at_index(self.vace_high, 0),
        )
        model_high_swap = wanvideosetblockswap_node.loadmodel(
            model=get_value_at_index(model_high_raw, 0),
            block_swap_args=get_value_at_index(self.blockswap_args, 0),
        )
        self.model_high = wanvideosetloras_node.setlora(
            model=get_value_at_index(model_high_swap, 0),
            lora=get_value_at_index(self.lora_multi_high, 0),
        )

        self.loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        self.loadimage.load_image = types.MethodType(
            support_pil_image(self.loadimage.load_image.__func__),
            self.loadimage,
        )

    @torch.inference_mode()
    def forward(self,image_1=None,image_2=None,prompt=None,num_frames=81,width=780,height=420,fps=24):
        """
        首帧图 + 尾帧图 + 文本 prompt -> WanVideo VACE 首尾帧视频，返回本地视频路径。
        """
        
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

        width_val = intconstant_node.get_value(value=width)
        height_val = intconstant_node.get_value(value=height)
        num_frames_val = intconstant_node.get_value(value=num_frames)
        fps_val = floatconstant_node.get_value(value=fps)

        load_start = self.loadimage.load_image(image=image_1)
        load_end = self.loadimage.load_image(image=image_2)
        resized_start = imageresizekjv2_node.resize(
            width=get_value_at_index(width_val, 0),
            height=get_value_at_index(height_val, 0),
            upscale_method="lanczos",
            keep_proportion="crop",
            pad_color="0, 0, 0",
            crop_position="center",
            divisible_by=16,
            device="cpu",
            image=get_value_at_index(load_start, 0),
            unique_id=3000438492122948674,
        )
        resized_end = imageresizekjv2_node.resize(
            width=get_value_at_index(width_val, 0),
            height=get_value_at_index(height_val, 0),
            upscale_method="lanczos",
            keep_proportion="crop",
            pad_color="0, 0, 0",
            crop_position="center",
            divisible_by=16,
            device="cpu",
            image=get_value_at_index(load_end, 0),
            unique_id=7012091250016467665,
        )
        vacestarttoendframe_out = wanvideovacestarttoendframe_node.process(
            num_frames=get_value_at_index(num_frames_val, 0),
            empty_frame_level=0.5,
            start_index=0,
            end_index=-1,
            start_image=get_value_at_index(resized_start, 0),
            end_image=get_value_at_index(resized_end, 0),
        )
        getimagesizeandcount_out = getimagesizeandcount_node.getsize(
            image=get_value_at_index(vacestarttoendframe_out, 0)
        )
        scheduler_first = wanvideoscheduler_node.process(
            scheduler="dpm++_sde",
            steps=get_value_at_index(int_node.to_int(Number="8"), 0),
            shift=8,
            start_step=0,
            end_step=get_value_at_index(self.sigmatostep, 0),
            unique_id=14314671872500839117,
        )
        scheduler_second = wanvideoscheduler_node.process(
            scheduler="dpm++_sde",
            steps=get_value_at_index(int_node.to_int(Number="8"), 0),
            shift=8,
            start_step=get_value_at_index(self.sigmatostep, 0),
            end_step=-1,
            unique_id=10273909465475146181,
        )
        wanvideovaceencode_out = wanvideovaceencode_node.process(
            width=get_value_at_index(getimagesizeandcount_out, 1),
            height=get_value_at_index(getimagesizeandcount_out, 2),
            num_frames=get_value_at_index(getimagesizeandcount_out, 3),
            strength=1,
            vace_start_percent=0,
            vace_end_percent=1,
            tiled_vae=False,
            vae=get_value_at_index(self.vae, 0),
            input_frames=get_value_at_index(vacestarttoendframe_out, 0),
            input_masks=get_value_at_index(vacestarttoendframe_out, 1),
        )
        wanvideotextencode_out = wanvideotextencode_node.process(
            positive_prompt=prompt,
            negative_prompt=negative_prompt,
            force_offload=True,
            use_disk_cache=False,
            device="gpu",
            t5=get_value_at_index(self.t5, 0),
        )
        wanvideosampler_high = wanvideosampler_node.process(
            steps=get_value_at_index(scheduler_first, 1),
            cfg=get_value_at_index(self.stringtofloatlist_66, 0),
            shift=get_value_at_index(scheduler_first, 2),
            seed=random.randint(1, 2**64),
            force_offload=True,
            scheduler=get_value_at_index(scheduler_first, 3),
            riflex_freq_index=0,
            denoise_strength=1,
            batched_cfg=False,
            rope_function="comfy",
            start_step=get_value_at_index(scheduler_first, 4),
            end_step=get_value_at_index(scheduler_first, 5),
            add_noise_to_samples=False,
            model=get_value_at_index(self.model_high, 0),
            image_embeds=get_value_at_index(wanvideovaceencode_out, 0),
            text_embeds=get_value_at_index(wanvideotextencode_out, 0),
            feta_args=get_value_at_index(self.enhanceavideo_args, 0),
            cache_args=get_value_at_index(self.easycache_args, 0),
            slg_args=get_value_at_index(self.slg_args, 0),
            experimental_args=get_value_at_index(self.experimentalargs_args, 0),
        )
        wanvideosampler_low = wanvideosampler_node.process(
            steps=get_value_at_index(scheduler_second, 1),
            cfg=1,
            shift=get_value_at_index(scheduler_second, 2),
            seed=random.randint(1, 2**64),
            force_offload=True,
            scheduler=get_value_at_index(scheduler_second, 3),
            riflex_freq_index=0,
            denoise_strength=1,
            batched_cfg=False,
            rope_function="comfy",
            start_step=get_value_at_index(scheduler_second, 4),
            end_step=get_value_at_index(scheduler_second, 5),
            add_noise_to_samples=False,
            model=get_value_at_index(self.model_low, 0),
            image_embeds=get_value_at_index(wanvideovaceencode_out, 0),
            text_embeds=get_value_at_index(wanvideotextencode_out, 0),
            samples=get_value_at_index(wanvideosampler_high, 0),
            feta_args=get_value_at_index(self.enhanceavideo_args, 0),
            cache_args=get_value_at_index(self.easycache_args, 0),
            slg_args=get_value_at_index(self.slg_args, 0),
        )
        wanvideodecode_out = wanvideodecode_node.decode(
            enable_vae_tiling=False,
            tile_x=272,
            tile_y=272,
            tile_stride_x=144,
            tile_stride_y=128,
            normalization="default",
            vae=get_value_at_index(self.vae, 0),
            samples=get_value_at_index(wanvideosampler_low, 0),
        )
        vhs_videocombine_out = vhs_videocombine_node.combine_video(
            frame_rate=get_value_at_index(fps_val, 0),
            loop_count=0,
            filename_prefix="wan2.2vace_first_end",
            format="video/h264-mp4",
            pix_fmt="yuv420p",
            crf=19,
            save_metadata=True,
            trim_to_audio=False,
            pingpong=False,
            save_output=True,
            images=get_value_at_index(wanvideodecode_out, 0),
            unique_id=15159582174570800378,
        )
        video = get_value_at_index(vhs_videocombine_out, 0)[1][-1]

        # 尽量释放中间结果所占用的显存/内存
        try:
            del (
                width_val,
                height_val,
                num_frames_val,
                fps_val,
                load_start,
                load_end,
                resized_start,
                resized_end,
                vacestarttoendframe_out,
                getimagesizeandcount_out,
                scheduler_first,
                scheduler_second,
                wanvideovaceencode_out,
                wanvideotextencode_out,
                wanvideosampler_high,
                wanvideosampler_low,
                wanvideodecode_out,
                vhs_videocombine_out,
            )
        except Exception:
            pass

        cleanup_memory()
        return video

