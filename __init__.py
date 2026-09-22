from .load_image_crop import LoadImageCrop
from .apply_linked_crop import ApplyLinkedCrop

NODE_CLASS_MAPPINGS = {
    "LoadImageCrop": LoadImageCrop,
    "ApplyLinkedCrop": ApplyLinkedCrop,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageCrop": "Load Image + Crop",
    "ApplyLinkedCrop": "Apply Linked Crop",
}

WEB_DIRECTORY = "./web"
