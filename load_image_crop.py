import os
import hashlib
import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import comfy
import node_helpers

from .crop_utils import compute_resolution, resize_image, resize_mask, UPSCALE_METHODS

try: # current import location
    from comfy_api.latest import InputImpl
except Exception:
    try: # older import location
        from comfy.api.latest import InputImpl
    except Exception:
        InputImpl = None

ASPECT_RATIOS = [
    "Original",
    "Free (Custom)",
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]


class LoadImageCrop:
    """Like the official Load Image node, plus:
    - a visual, mouse-draggable crop area (aspect-ratio locked) in the frontend preview
    - a paste-from-clipboard entry in the node context menu

    With aspect_ratio == "Original" (default) the image is returned as-is,
    uncropped.

    Otherwise the crop rectangle (crop_x, crop_y, crop_w, crop_h) arrives as
    normalized [0..1] fractions of the source image size, computed by the
    frontend.
    """

    CATEGORY = "image"
    SEARCH_ALIASES = ["load image", "crop", "crop image", "paste", "clipboard"]
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "INT", "INT", "CROP_DATA")
    RETURN_NAMES = ("raw_image", "raw_mask", "resized_image", "resized_mask", "width", "height", "crop_data")
    FUNCTION = "load_image"
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "aspect_ratio": (ASPECT_RATIOS,),
                "crop_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "crop_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "crop_w": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "crop_h": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "multiple": ("INT", {"default": 8, "min": 8, "max": 128, "step": 4}),
                "upscale_method": (UPSCALE_METHODS, {"default": "lanczos"}),
            }
        }

    def load_image(self, image, aspect_ratio, crop_x, crop_y, crop_w, crop_h,
                    megapixels, multiple, upscale_method):
        output_image, output_mask = self._load(image)

        if aspect_ratio == "Original":
            crop_x, crop_y, crop_w, crop_h = 0.0, 0.0, 1.0, 1.0  # no stale crop leaks into crop_data
        else:
            output_image, output_mask = self._crop(output_image, output_mask, crop_x, crop_y, crop_w, crop_h)

        H, W = output_image.shape[1], output_image.shape[2]
        ratio = W / H
        width, height = compute_resolution(ratio, megapixels, multiple)

        resized_image = resize_image(output_image, width, height, upscale_method)
        resized_mask = resize_mask(output_mask, width, height, upscale_method)

        crop_data = {
            "crop_x": crop_x, "crop_y": crop_y, "crop_w": crop_w, "crop_h": crop_h,
            "width": width, "height": height, "aspect_ratio": aspect_ratio,
        }

        return (output_image, output_mask, resized_image, resized_mask, width, height, crop_data)

    def _load(self, image):
        """Load the image (and its alpha mask) as tensors, without any crop."""
        image_path = folder_paths.get_annotated_filepath(image)
        dtype = comfy.model_management.intermediate_dtype()
        device = comfy.model_management.intermediate_device()

        if InputImpl is not None:
            components = InputImpl.VideoFromFile(image_path).get_components()
            if components.images.shape[0] > 0:
                output_image = components.images.to(device=device, dtype=dtype)
                if components.alpha is not None:
                    output_mask = (1.0 - components.alpha[..., -1]).to(device=device, dtype=dtype)
                else:
                    output_mask = torch.zeros((output_image.shape[0], 64, 64), dtype=dtype, device=device)
                return (output_image, output_mask)

        # Fallback (also used when InputImpl is unavailable): plain image loading,
        # the same path the official node uses for animated webp.
        img = node_helpers.pillow(Image.open, image_path)
        output_images, output_masks = [], []
        w, h = None, None
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            rgb = i.convert("RGB")
            if len(output_images) == 0:
                w, h = rgb.size[0], rgb.size[1]
            if rgb.size[0] != w or rgb.size[1] != h:
                continue
            arr = np.array(rgb).astype(np.float32) / 255.0
            output_images.append(torch.from_numpy(arr)[None,].to(dtype=dtype))
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32)
            output_masks.append(mask.unsqueeze(0).to(dtype=dtype))

        if len(output_images) == 0:
            raise RuntimeError("no usable frames in image: {}".format(image))

        output_image = torch.cat(output_images, dim=0).to(device=device, dtype=dtype)
        output_mask = torch.cat(output_masks, dim=0).to(device=device, dtype=dtype)
        return (output_image, output_mask)

    def _crop(self, output_image, output_mask, crop_x, crop_y, crop_w, crop_h):
        # full-frame crop ("Original" mode): return as-is
        if (crop_x, crop_y, crop_w, crop_h) == (0, 0, 1, 1):
            return (output_image, output_mask)

        # guard against degenerate values (API/workflow input): an empty or
        # out-of-range crop is returned uncropped
        if not (crop_w > 0.001 and crop_h > 0.001 and crop_x < 0.999 and crop_y < 0.999):
            return (output_image, output_mask)

        H, W = output_image.shape[1], output_image.shape[2]
        mh, mw = output_mask.shape[1], output_mask.shape[2]

        def clampi(v, lo, hi):
            return max(lo, min(hi, int(round(v))))

        # crop the image
        x0 = clampi(crop_x * W, 0, W - 1)
        y0 = clampi(crop_y * H, 0, H - 1)
        x1 = clampi((crop_x + crop_w) * W, x0 + 1, W)
        y1 = clampi((crop_y + crop_h) * H, y0 + 1, H)
        output_image = output_image[:, y0:y1, x0:x1, :]

        # crop the mask proportionally (its size may differ from the image size)
        mx0 = clampi(crop_x * mw, 0, mw - 1)
        my0 = clampi(crop_y * mh, 0, mh - 1)
        mx1 = clampi((crop_x + crop_w) * mw, mx0 + 1, mw)
        my1 = clampi((crop_y + crop_h) * mh, my0 + 1, mh)
        output_mask = output_mask[:, my0:my1, mx0:mx1]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image, aspect_ratio, crop_x, crop_y, crop_w, crop_h, megapixels, multiple, upscale_method):
        # file identity via stat (size + mtime) instead of hashing the whole file
        image_path = folder_paths.get_annotated_filepath(image)
        st = os.stat(image_path)
        m = hashlib.sha256()
        m.update(str(st.st_size).encode())
        m.update(str(st.st_mtime_ns).encode())
        # full-precision crop + resize values so any change forces a re-run
        m.update(f"|{aspect_ratio}|{crop_x!r},{crop_y!r},{crop_w!r},{crop_h!r}|{megapixels!r},{multiple!r},{upscale_method}".encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image, aspect_ratio=None, crop_x=None, crop_y=None, crop_w=None, crop_h=None,
                         megapixels=None, multiple=None, upscale_method=None):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
