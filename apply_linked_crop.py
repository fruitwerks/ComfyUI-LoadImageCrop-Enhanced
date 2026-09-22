from .crop_utils import (
    crop_image, crop_mask, resize_image, resize_mask,
    UPSCALE_METHODS, MASK_RESIZE_METHODS,
)


class ApplyLinkedCrop:
    """Applies a crop rectangle + target resolution produced by LoadImageCrop's
    crop_data output to a second image and/or mask, so a reference image and a
    paired mask (or second image) always end up framed and sized identically.
    """

    CATEGORY = "image"
    SEARCH_ALIASES = ["crop", "mask", "linked crop", "apply crop"]
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("raw_image", "resized_image", "raw_mask", "resized_mask")
    FUNCTION = "apply"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "crop_data": ("CROP_DATA",),
                "upscale_method": (UPSCALE_METHODS, {"default": "lanczos"}),
                "match_image_resize": ("BOOLEAN", {"default": False}),
                "mask_resize_method": (MASK_RESIZE_METHODS, {"default": "bilinear"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    def apply(self, crop_data, upscale_method, match_image_resize, mask_resize_method,
              image=None, mask=None):
        crop_x, crop_y = crop_data["crop_x"], crop_data["crop_y"]
        crop_w, crop_h = crop_data["crop_w"], crop_data["crop_h"]
        width, height = crop_data["width"], crop_data["height"]

        raw_image = resized_image = None
        raw_mask = resized_mask = None

        if image is not None:
            raw_image = crop_image(image, crop_x, crop_y, crop_w, crop_h)
            resized_image = resize_image(raw_image, width, height, upscale_method)

        if mask is not None:
            raw_mask = crop_mask(mask, crop_x, crop_y, crop_w, crop_h)
            method = upscale_method if match_image_resize else mask_resize_method
            resized_mask = resize_mask(raw_mask, width, height, method)

        return (raw_image, resized_image, raw_mask, resized_mask)
