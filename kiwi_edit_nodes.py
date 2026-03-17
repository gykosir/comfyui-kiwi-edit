import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import comfy.model_management as model_management
except ImportError:  # pragma: no cover - fallback for local editing outside ComfyUI
    model_management = None

try:
    import folder_paths
except ImportError:  # pragma: no cover - fallback for local editing outside ComfyUI
    folder_paths = None


_PIPELINE_CACHE: Dict[Tuple[str, str, torch.dtype], Any] = {}


def _require_diffusers() -> Tuple[Any, Any]:
    try:
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video
    except ImportError as error:
        raise RuntimeError(
            "Kiwi Edit requires a compatible diffusers stack at runtime. "
            "Install the packages from requirements.txt and ensure peft>=0.17.0."
        ) from error
    return DiffusionPipeline, export_to_video


def _resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    if model_management is not None:
        return str(model_management.get_torch_device())
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16


def _get_pipeline(model_path: str, device_name: str, dtype_name: str) -> Any:
    device = _resolve_device(device_name)
    dtype = _resolve_dtype(dtype_name)
    cache_key = (model_path, device, dtype)
    pipeline = _PIPELINE_CACHE.get(cache_key)
    if pipeline is None:
        DiffusionPipeline, _ = _require_diffusers()
        pipeline = DiffusionPipeline.from_pretrained(model_path, trust_remote_code=True)
        pipeline.to(device, dtype=dtype)
        _PIPELINE_CACHE[cache_key] = pipeline
    return pipeline


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _resize_to_multiple_of_16(image: Image.Image) -> Image.Image:
    width, height = image.size
    target_width = max(16, (width // 16) * 16)
    target_height = max(16, (height // 16) * 16)
    if (target_width, target_height) == (width, height):
        return image
    return image.resize((target_width, target_height), Image.LANCZOS)


def _tensor_to_pil_list(
    images: torch.Tensor, resize_to_multiple_of_16: bool
) -> Tuple[List[Image.Image], Tuple[int, int]]:
    if images.ndim == 3:
        images = images.unsqueeze(0)
    images = images.detach().cpu().clamp(0.0, 1.0)
    original_height, original_width = int(images.shape[1]), int(images.shape[2])
    pil_images: List[Image.Image] = []
    for frame in images:
        array = (frame.numpy() * 255.0).round().astype(np.uint8)
        image = _ensure_rgb(Image.fromarray(array))
        if resize_to_multiple_of_16:
            image = _resize_to_multiple_of_16(image)
        pil_images.append(image)
    return pil_images, (original_width, original_height)


def _pil_to_tensor(
    images: List[Image.Image], target_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    tensors: List[torch.Tensor] = []
    for image in images:
        current = _ensure_rgb(image)
        if target_size is not None and current.size != target_size:
            current = current.resize(target_size, Image.LANCZOS)
        array = np.asarray(current).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(array))
    return torch.stack(tensors, dim=0)


def _prepare_ref_image(
    ref_image: Optional[torch.Tensor], target_size: Tuple[int, int]
) -> Optional[List[Image.Image]]:
    if ref_image is None:
        return None
    if ref_image.ndim == 3:
        ref_image = ref_image.unsqueeze(0)
    first = ref_image[:1].detach().cpu().clamp(0.0, 1.0)
    image = Image.fromarray(
        (first[0].numpy() * 255.0).round().astype(np.uint8)
    ).convert("RGB")
    if image.size != target_size:
        image = image.resize(target_size, Image.LANCZOS)
    return [image]


def _load_video_frames(
    video_path: str, max_frames: int, max_pixels: int
) -> torch.Tensor:
    from torchvision.io import read_video

    video_frames, _, _ = read_video(video_path, pts_unit="sec")
    frames: List[Image.Image] = []
    for index in range(min(len(video_frames), max_frames)):
        image = Image.fromarray(video_frames[index].numpy())
        width, height = image.size
        scale = min(1.0, (max_pixels / (width * height)) ** 0.5)
        if scale < 1.0:
            resized_width = max(16, int(width * scale) // 16 * 16)
            resized_height = max(16, int(height * scale) // 16 * 16)
            image = image.resize((resized_width, resized_height), Image.LANCZOS)
        frames.append(image.convert("RGB"))
    if not frames:
        raise ValueError(f"No frames could be loaded from video: {video_path}")
    return _pil_to_tensor(frames)


def _sanitize_filename_part(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in ("-", "_") else "_"
        for character in value.strip()
    )
    return cleaned.strip("._") or "kiwi_edit"


def _resolve_output_path(filename_prefix: str, file_stem: str) -> Path:
    if filename_prefix.strip():
        prefix_path = Path(os.path.expanduser(filename_prefix.strip()))
    else:
        prefix_path = Path(file_stem)

    has_suffix = prefix_path.suffix.lower() == ".mp4"
    if prefix_path.is_absolute():
        target = prefix_path if has_suffix else prefix_path.with_suffix(".mp4")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    if folder_paths is not None:
        output_directory = Path(folder_paths.get_output_directory())
    else:
        output_directory = Path.cwd() / "output"

    relative_target = prefix_path if has_suffix else prefix_path.with_suffix(".mp4")
    target = output_directory / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)

    if not target.exists():
        return target

    stem = _sanitize_filename_part(target.stem)
    suffix = target.suffix or ".mp4"
    counter = 1
    while True:
        candidate = target.with_name(f"{stem}_{counter:04d}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


class KiwiEditLoadVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "max_frames": ("INT", {"default": 81, "min": 1, "max": 256, "step": 1}),
                "max_pixels": (
                    "INT",
                    {
                        "default": 720 * 1280,
                        "min": 16 * 16,
                        "max": 4096 * 4096,
                        "step": 16,
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "load"
    CATEGORY = "KiwiEdit"

    def load(self, video_path: str, max_frames: int, max_pixels: int):
        normalized_path = os.path.expanduser(video_path.strip())
        if not normalized_path or not os.path.exists(normalized_path):
            raise FileNotFoundError(f"Video path does not exist: {video_path}")
        frames = _load_video_frames(normalized_path, max_frames, max_pixels)
        return (frames,)


class KiwiEditVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model_path": (
                    "STRING",
                    {
                        "default": "linyq/kiwi-edit-5b-instruct-reference-diffusers",
                        "multiline": False,
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {"default": 50, "min": 1, "max": 200, "step": 1},
                ),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "device": (
                    ["auto", "cuda", "cuda:0", "cuda:1", "cpu"],
                    {"default": "auto"},
                ),
                "dtype": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "tiled": ("BOOLEAN", {"default": True}),
                "resize_to_multiple_of_16": ("BOOLEAN", {"default": True}),
                "clear_cache_after_run": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "ref_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate"
    CATEGORY = "KiwiEdit"

    def generate(
        self,
        frames: torch.Tensor,
        prompt: str,
        model_path: str,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        device: str,
        dtype: str,
        tiled: bool,
        resize_to_multiple_of_16: bool,
        clear_cache_after_run: bool,
        ref_image: Optional[torch.Tensor] = None,
    ):
        source_frames, original_size = _tensor_to_pil_list(
            frames, resize_to_multiple_of_16
        )
        width, height = source_frames[0].size
        pipeline = _get_pipeline(model_path, device, dtype)
        prepared_ref_image = _prepare_ref_image(ref_image, (width, height))

        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                source_video=source_frames,
                ref_image=prepared_ref_image,
                height=height,
                width=width,
                num_frames=len(source_frames),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=int(seed),
                tiled=bool(tiled),
            )

        if hasattr(result, "frames"):
            generated_frames = (
                result.frames[0]
                if result.frames and isinstance(result.frames[0], list)
                else result.frames
            )
        elif hasattr(result, "images"):
            generated_frames = result.images
        else:
            generated_frames = result

        if not isinstance(generated_frames, list) or not generated_frames:
            raise TypeError("Pipeline output format is not supported by this node.")

        output = _pil_to_tensor(generated_frames, target_size=original_size)
        if clear_cache_after_run:
            if model_management is not None:
                model_management.soft_empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        return (output,)


class KiwiEditSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "filename_prefix": (
                    "STRING",
                    {"default": "KiwiEdit/kiwi_edit_output", "multiline": False},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 15.0, "min": 1.0, "max": 120.0, "step": 0.5},
                ),
                "resize_to_multiple_of_16": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "KiwiEdit"

    def save(
        self,
        frames: torch.Tensor,
        filename_prefix: str,
        fps: float,
        resize_to_multiple_of_16: bool,
    ):
        _, export_to_video = _require_diffusers()
        pil_frames, _ = _tensor_to_pil_list(frames, resize_to_multiple_of_16)
        output_path = _resolve_output_path(filename_prefix, "kiwi_edit_output")
        export_to_video(pil_frames, str(output_path), fps=float(fps))
        return {
            "ui": {"text": [str(output_path)]},
            "result": (str(output_path),),
        }


NODE_CLASS_MAPPINGS = {
    "KiwiEditLoadVideo": KiwiEditLoadVideo,
    "KiwiEditSaveVideo": KiwiEditSaveVideo,
    "KiwiEditVideoNode": KiwiEditVideoNode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "KiwiEditLoadVideo": "Kiwi Edit Load Video",
    "KiwiEditSaveVideo": "Kiwi Edit Save Video",
    "KiwiEditVideoNode": "Kiwi Edit Video",
}
