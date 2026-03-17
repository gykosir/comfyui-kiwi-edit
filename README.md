# Kiwi Edit ComfyUI Nodes

ComfyUI custom nodes for running the Kiwi Edit diffusers pipeline on a video frame batch.

The nodes appear under the top-level `KiwiEdit` category.

## Included nodes

- `Kiwi Edit Load Video`
  - Loads a video file from disk.
  - Returns the frames as a ComfyUI `IMAGE` batch.
- `Kiwi Edit Video`
  - Runs the `linyq/kiwi-edit-5b-instruct-reference-diffusers` pipeline.
  - Accepts an `IMAGE` batch as input.
  - Supports an optional `ref_image` input.
- `Kiwi Edit Save Video`
  - Saves an `IMAGE` batch to MP4.
  - Writes into the ComfyUI output directory by default.

## Installation

1. Place this folder inside `ComfyUI/custom_nodes/`.
2. Install the dependencies into the same Python environment that ComfyUI uses.
3. Restart ComfyUI.

Example on this Windows setup:

```powershell
g:\comfy\python_embeded\python.exe -m pip install -r requirements.txt
```

If you only need to fix the known dependency conflict:

```powershell
g:\comfy\python_embeded\python.exe -m pip install -U "peft>=0.17.0"
```

## Usage

1. Add `Kiwi Edit Load Video` and point it to an input video.
2. Connect its `frames` output to `Kiwi Edit Video`.
3. Enter your edit prompt.
4. Save the generated result with `Kiwi Edit Save Video`.

If you already use another loader that outputs an `IMAGE` batch, you can connect that directly to `Kiwi Edit Video` and skip `Kiwi Edit Load Video`.

## Notes

- The model is loaded lazily when `Kiwi Edit Video` or `Kiwi Edit Save Video` runs.
- This means the node pack can still appear in ComfyUI even if the diffusers runtime stack is not fully usable yet.
- If the runtime dependencies are incomplete, the node will fail when executed rather than disappearing from the node list.

## Troubleshooting

### Node does not appear

Check `ComfyUI/user/comfyui.log` for an import failure.

Common causes:

- `peft` is too old for the installed `diffusers` version.
- Dependencies were installed into a different Python environment than the one ComfyUI is using.
- ComfyUI was not restarted after changing the node files.

This ComfyUI install is using:

- Python executable: `g:\comfy\python_embeded\python.exe`
- Log file: `g:\comfy\ComfyUI\user\comfyui.log`

### Node appears but execution fails

Usually this means the node pack loaded successfully, but the runtime dependencies are still missing or incompatible.

Reinstall the requirements with the ComfyUI Python executable:

```powershell
g:\comfy\python_embeded\python.exe -m pip install -r requirements.txt
```

### I cannot find the nodes in the menu

Search for `Kiwi` in the Add Node dialog, or browse the top-level `KiwiEdit` category.