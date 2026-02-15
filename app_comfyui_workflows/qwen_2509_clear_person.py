import os
import random
import sys
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
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
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
    comfyui_path = find_path("ComfyUI")
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

    sys.path.insert(0, find_path("ComfyUI"))
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    asyncio.run(init_extra_nodes())


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_20 = unetloader.load_unet(
            unet_name="qwen/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
            weight_dtype="fp8_e4m3fn",
        )

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_21 = loraloadermodelonly.load_lora_model_only(
            lora_name="qwen/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors",
            strength_model=1,
            model=get_value_at_index(unetloader_20, 0),
        )

        loraloadermodelonly_4 = loraloadermodelonly.load_lora_model_only(
            lora_name="qwen/光影渲染产品溶图Qwen-Edit_2509.safetensors",
            strength_model=1,
            model=get_value_at_index(loraloadermodelonly_21, 0),
        )

        text_multiline = NODE_CLASS_MAPPINGS["Text Multiline"]()
        text_multiline_8 = text_multiline.text_multiline(text="消除图一中的人物")

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_22 = vaeloader.load_vae(vae_name="qwen_image_vae.safetensors")

        cliploader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
        cliploader_23 = cliploader.load_clip(
            clip_name="qwen_2.5_vl_7b_fp8_scaled.safetensors",
            type="qwen_image",
            device="default",
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_24 = loadimage.load_image(image="11.jpg")

        textencodeqwenimageeditplus = NODE_CLASS_MAPPINGS[
            "TextEncodeQwenImageEditPlus"
        ]()
        conditioningzeroout = NODE_CLASS_MAPPINGS["ConditioningZeroOut"]()
        get_image_size = NODE_CLASS_MAPPINGS["Get Image Size"]()
        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        for q in range(1):
            textencodeqwenimageeditplus_7 = (
                textencodeqwenimageeditplus.EXECUTE_NORMALIZED(
                    prompt=get_value_at_index(text_multiline_8, 0),
                    clip=get_value_at_index(cliploader_23, 0),
                    vae=get_value_at_index(vaeloader_22, 0),
                    image1=get_value_at_index(loadimage_24, 0),
                )
            )

            conditioningzeroout_3 = conditioningzeroout.zero_out(
                conditioning=get_value_at_index(textencodeqwenimageeditplus_7, 0)
            )

            get_image_size_10 = get_image_size.get_size(
                image=get_value_at_index(loadimage_24, 0)
            )

            emptylatentimage_11 = emptylatentimage.generate(
                width=get_value_at_index(get_image_size_10, 0),
                height=get_value_at_index(get_image_size_10, 1),
                batch_size=1,
            )

            ksampler_12 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=10,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                denoise=1,
                model=get_value_at_index(loraloadermodelonly_4, 0),
                positive=get_value_at_index(textencodeqwenimageeditplus_7, 0),
                negative=get_value_at_index(conditioningzeroout_3, 0),
                latent_image=get_value_at_index(emptylatentimage_11, 0),
            )

            vaedecode_15 = vaedecode.decode(
                samples=get_value_at_index(ksampler_12, 0),
                vae=get_value_at_index(vaeloader_22, 0),
            )


if __name__ == "__main__":
    main()
