import math
import comfy.utils

UPSCALE_METHODS = ["lanczos", "bicubic", "bilinear"]
MASK_RESIZE_METHODS = ["bilinear", "lanczos", "bicubic"]


def compute_resolution(ratio, megapixels, multiple):
    """ratio = width / height. Returns (width, height), each a multiple of `multiple`."""
    target_px = megapixels * 1024 * 1024
    h = math.sqrt(target_px / ratio)
    w = h * ratio

    def snap(x, m):
        return max(m, int(round(x / m)) * m)

    return snap(w, multiple), snap(h, multiple)


def _clampi(v, lo, hi):
    return max(lo, min(hi, int(round(v))))


def crop_image(image, crop_x, crop_y, crop_w, crop_h):
    """image: (B,H,W,C). crop_* normalized 0..1."""
    if (crop_x, crop_y, crop_w, crop_h) == (0, 0, 1, 1):
        return image
    if not (crop_w > 0.001 and crop_h > 0.001 and crop_x < 0.999 and crop_y < 0.999):
        return image
    H, W = image.shape[1], image.shape[2]
    x0 = _clampi(crop_x * W, 0, W - 1)
    y0 = _clampi(crop_y * H, 0, H - 1)
    x1 = _clampi((crop_x + crop_w) * W, x0 + 1, W)
    y1 = _clampi((crop_y + crop_h) * H, y0 + 1, H)
    return image[:, y0:y1, x0:x1, :]


def crop_mask(mask, crop_x, crop_y, crop_w, crop_h):
    """mask: (B,H,W). crop_* normalized 0..1 (computed independently of mask's own res)."""
    if (crop_x, crop_y, crop_w, crop_h) == (0, 0, 1, 1):
        return mask
    if not (crop_w > 0.001 and crop_h > 0.001 and crop_x < 0.999 and crop_y < 0.999):
        return mask
    H, W = mask.shape[1], mask.shape[2]
    x0 = _clampi(crop_x * W, 0, W - 1)
    y0 = _clampi(crop_y * H, 0, H - 1)
    x1 = _clampi((crop_x + crop_w) * W, x0 + 1, W)
    y1 = _clampi((crop_y + crop_h) * H, y0 + 1, H)
    return mask[:, y0:y1, x0:x1]


def resize_image(image, width, height, method):
    """image: (B,H,W,C) -> (B,height,width,C), via native comfy.utils.common_upscale."""
    samples = image.movedim(-1, 1)
    scaled = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return scaled.movedim(1, -1)


def resize_mask(mask, width, height, method):
    """mask: (B,H,W) -> (B,height,width), clamped to [0,1] to guard against Lanczos/bicubic overshoot."""
    samples = mask.unsqueeze(1)
    scaled = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return scaled.squeeze(1).clamp(0.0, 1.0)
