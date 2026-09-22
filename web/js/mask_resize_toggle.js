import { app } from "../../scripts/app.js";

app.registerExtension({
  name: "ComfyUI.LoadImageCrop.MaskResizeToggle",
  async nodeCreated(node) {
    if (node.comfyClass !== "ApplyLinkedCrop") return;

    const toggle = node.widgets.find(w => w.name === "match_image_resize");
    const maskMethod = node.widgets.find(w => w.name === "mask_resize_method");
    if (!toggle || !maskMethod) return;

    const sync = () => {
      maskMethod.disabled = !!toggle.value;
      node.setDirtyCanvas(true, true);
    };

    const origCallback = toggle.callback;
    toggle.callback = function (...args) {
      origCallback?.apply(this, args);
      sync();
    };

    sync();
  },
});
