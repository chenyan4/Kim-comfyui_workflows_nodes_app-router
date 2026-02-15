try:
    from .nodes import *
    NODE_CONFIG = {
        #constants
        # mask_subtraction_node
        #conditioning
        # "CondPassThrough": {"class": CondPassThrough},
        #masking

    }

    __all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

except Exception as e:
    print(f"## zdx_comfyui nodes failed: {e}")
    pass