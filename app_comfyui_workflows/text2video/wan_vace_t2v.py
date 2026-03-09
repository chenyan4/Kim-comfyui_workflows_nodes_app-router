import os
import random
import sys
import gc
from typing import Sequence, Mapping, Any, Union
import torch


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

# 节点实例（与 Zimage_text2img 风格一致，模块级缓存）
wanvideovacemodelselect_node = NODE_CLASS_MAPPINGS["WanVideoVACEModelSelect"]()
wanvideoloraselect_node = NODE_CLASS_MAPPINGS["WanVideoLoraSelect"]()
wanvideoblockswap_node = NODE_CLASS_MAPPINGS["WanVideoBlockSwap"]()
loadwanvideot5textencoder_node = NODE_CLASS_MAPPINGS["LoadWanVideoT5TextEncoder"]()
wanvideovaeloader_node = NODE_CLASS_MAPPINGS["WanVideoVAELoader"]()
wanvideomodelloader_high_node = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
wanvideomodelloader_low_node = NODE_CLASS_MAPPINGS["WanVideoModelLoader"]()
int_node = NODE_CLASS_MAPPINGS["Int"]()
intconstant_node = NODE_CLASS_MAPPINGS["INTConstant"]()
floatconstant_node = NODE_CLASS_MAPPINGS["FloatConstant"]()
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


class WanVaceT2V:
    def __init__(self):
        # 预加载模型，风格对齐 Zimage_text2img
        self.vace_high = wanvideovacemodelselect_node.getvacepath(
            vace_model="Wan2.2_VACE/Wan2_2_Fun_VACE_module_A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
        )
        self.vace_low = wanvideovacemodelselect_node.getvacepath(
            vace_model="Wan2.2_VACE/Wan2_2_Fun_VACE_module_A14B_LOW_fp8_e4m3fn_scaled_KJ.safetensors"
        )
        self.lora_high = wanvideoloraselect_node.getlorapath(
            lora="Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
            strength=1,
            low_mem_load=False,
            merge_loras=True,
            unique_id=7783083932390163562,
        )
        self.lora_low = wanvideoloraselect_node.getlorapath(
            lora="Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
            strength=1,
            low_mem_load=False,
            merge_loras=True,
            unique_id=1728372192399379055,
        )
        self.blockswap_args = wanvideoblockswap_node.setargs(
            blocks_to_swap=25,
            offload_img_emb=False,
            offload_txt_emb=False,
            use_non_blocking=True,
            vace_blocks_to_swap=0,
            prefetch_blocks=0,
            block_swap_debug=False,
        )
        self.t5 = loadwanvideot5textencoder_node.loadmodel(
            model_name="umt5_xxl_fp16.safetensors",
            precision="bf16",
            load_device="offload_device",
            quantization="disabled",
        )
        self.vae = wanvideovaeloader_node.loadmodel(
            model_name="wan_2.1_vae.safetensors", precision="bf16", use_cpu_cache=True
        )
        self.model_high = wanvideomodelloader_high_node.loadmodel(
            model="Wan2.2/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
            base_precision="fp16_fast",
            quantization="disabled",
            load_device="offload_device",
            attention_mode="sageattn",
            rms_norm_function="default",
            block_swap_args=get_value_at_index(self.blockswap_args, 0),
            lora=get_value_at_index(self.lora_high, 0),
            extra_model=get_value_at_index(self.vace_high, 0),
        )
        self.model_low = wanvideomodelloader_low_node.loadmodel(
            model="Wan2.2/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
            base_precision="fp16_fast",
            quantization="disabled",
            load_device="offload_device",
            attention_mode="sageattn",
            rms_norm_function="default",
            block_swap_args=get_value_at_index(self.blockswap_args, 0),
            lora=get_value_at_index(self.lora_low, 0),
            extra_model=get_value_at_index(self.vace_low, 0),
        )

    @torch.inference_mode()
    def forward(
        self,
        prompt: str,
        width: int = 420,
        height: int = 780,
        num_frames: int = 101,
        fps: float = 24,
    ):
        """
        文本 prompt -> WanVideo VACE 文生视频后的输出（VHS_VideoCombine 的 images 等）。
        """
        negative_prompt = DEFAULT_NEGATIVE_PROMPT

        width_val = intconstant_node.get_value(value=width)
        height_val = intconstant_node.get_value(value=height)
        num_frames_val = intconstant_node.get_value(value=num_frames)
        fps_val = floatconstant_node.get_value(value=fps)

        wanvideovaceencode_out = wanvideovaceencode_node.process(
            width=get_value_at_index(width_val, 0),
            height=get_value_at_index(height_val, 0),
            num_frames=get_value_at_index(num_frames_val, 0),
            strength=1,
            vace_start_percent=0,
            vace_end_percent=1,
            tiled_vae=False,
            vae=get_value_at_index(self.vae, 0),
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
            steps=8,
            cfg=1,
            shift=5,
            seed=random.randint(1, 2**64),
            force_offload=True,
            scheduler="dpm++_sde",
            riflex_freq_index=0,
            denoise_strength=1,
            batched_cfg=False,
            rope_function="comfy",
            start_step=0,
            end_step=4,
            add_noise_to_samples=False,
            model=get_value_at_index(self.model_high, 0),
            image_embeds=get_value_at_index(wanvideovaceencode_out, 0),
            text_embeds=get_value_at_index(wanvideotextencode_out, 0),
        )
        wanvideosampler_low = wanvideosampler_node.process(
            steps=8,
            cfg=1,
            shift=5,
            seed=random.randint(1, 2**64),
            force_offload=True,
            scheduler="dpm++_sde",
            riflex_freq_index=0,
            denoise_strength=1,
            batched_cfg=False,
            rope_function="comfy",
            start_step=4,
            end_step=-1,
            add_noise_to_samples=False,
            model=get_value_at_index(self.model_low, 0),
            image_embeds=get_value_at_index(wanvideovaceencode_out, 0),
            text_embeds=get_value_at_index(wanvideotextencode_out, 0),
            samples=get_value_at_index(wanvideosampler_high, 0),
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
            filename_prefix="Wan_t2v",
            format="video/h264-mp4",
            pix_fmt="yuv420p",
            crf=19,
            save_metadata=True,
            trim_to_audio=True,
            pingpong=False,
            save_output=True,
            images=get_value_at_index(wanvideodecode_out, 0),
            unique_id=10353144037217341364,
        )

        video = get_value_at_index(vhs_videocombine_out, 0)[1][-1]

        # 尽量释放中间结果所占用的显存/内存
        try:
            del (
                width_val,
                height_val,
                num_frames_val,
                fps_val,
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



