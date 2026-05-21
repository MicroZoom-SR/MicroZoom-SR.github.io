import os
import glob
import shutil
import torch
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from diffusers import FluxControlNetModel
from diffusers.utils import convert_unet_state_dict_to_peft
import argparse
import glob
import time
import mediapy
import re
from datetime import datetime
import psutil  
from pathlib import Path

import sys
import deepzoom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from microzoom.pipelines.flux.custom_multidiffusion_pipeline_flux_controlnet_offload_v10 import FluxControlNetMultiDiffusionPipeline
from peft import set_peft_model_state_dict


global_start_time = time.time()
CONTROLNET_LORA_WEIGHTS_NAME = "controlnet_lora_layers.pt"

# 0. helper functions
from torchvision import transforms
def pil_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),                   # Convert to tensor in range [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5),   # Normalize to range [-1, 1]
                             (0.5, 0.5, 0.5))
    ])
    tensor = transform(img).unsqueeze(0)        # Add batch dimension (1, 3, H, W)
    return tensor


# 1. load params
parser = argparse.ArgumentParser()
parser.add_argument("--data_name", type=str, required=True)
parser.add_argument("--exp_name", type=str, default=None)
parser.add_argument("--ckpt_name", type=str, default="checkpoint-ep0001")
parser.add_argument("--lora_root", type=str, default=None)
parser.add_argument("--test_path", type=str, default=None)
parser.add_argument("--test_dir", type=str, default=None)
parser.add_argument("--scale_path", type=str, required=True)
parser.add_argument("--datetime_str", type=str, default=None)
parser.add_argument("--use_pretrained", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--debug_oom", action="store_true")
parser.add_argument("--use_gaussian_mask", action="store_true")
parser.add_argument("--sigma", type=int, default=256)
parser.add_argument("--debug_size", type=int, default=2048)
parser.add_argument("--res", type=int, default=1024)
parser.add_argument("--num_inference_steps", type=int, default=None)
parser.add_argument("--model_type", type=str, default="black-forest-labs/FLUX.1-dev", choices=["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-Krea-dev", "black-forest-labs/FLUX.1-dev"])
parser.add_argument("--delete_prev", action="store_true")
parser.add_argument("--chunk_size", type=int, default=2500)
parser.add_argument("--stride_method", type=str, default="constant", choices=["constant", "linear", "random"])
parser.add_argument("--remask", action="store_true")  # skip inference, just remask the output
parser.add_argument("--ablation_name", type=str, default=None)
parser.add_argument("--prompt_path", type=str, default=None)
parser.add_argument("--label_map_path", type=str, default=None)
parser.add_argument("--num_textures", type=int, default=3)
parser.add_argument("--write_dzi", action="store_true")
parser.add_argument("--stage_result_path", type=str, default=None)
parser.add_argument("--stage_mode", type=str, default="symlink", choices=["symlink", "copy"])
args = parser.parse_args()


def _resolve_single_path(paths, description):
  if len(paths) != 1:
    raise ValueError(f"Expected exactly one {description}, found {len(paths)}: {paths}")
  return paths[0]


def _resolve_test_path(test_path, test_dir):
  if test_path is not None:
    return test_path
  if test_dir is None:
    raise ValueError("Must provide either --test_path or --test_dir")

  candidates = sorted(glob.glob(os.path.join(test_dir, "*_close*.jpg")))
  candidates = [path for path in candidates if "_label_map" not in os.path.basename(path)]
  return _resolve_single_path(candidates, "held-out close-up JPG")


def _resolve_label_map_path(label_map_path, test_dir):
  if label_map_path is not None:
    return label_map_path
  if test_dir is None:
    raise ValueError("Must provide either --label_map_path or --test_dir")

  candidates = sorted(glob.glob(os.path.join(test_dir, "*_close*_label_map.png")))
  return _resolve_single_path(candidates, "held-out label map")


def _resolve_prompt_path(prompt_path, test_dir):
  if prompt_path is not None:
    return prompt_path
  if test_dir is None:
    raise ValueError("Must provide either --prompt_path or --test_dir")

  candidate = os.path.join(test_dir, "prompts.txt")
  if not os.path.isfile(candidate):
    raise ValueError(f"Prompt file not found: {candidate}")
  return candidate


def _resolve_mask_path(test_path):
  test_path_obj = Path(test_path)
  candidates = sorted(test_path_obj.parent.glob(f"{test_path_obj.stem}_mask*.png"))
  if len(candidates) > 1:
    raise ValueError(f"Expected at most one mask for {test_path}, found {len(candidates)}: {candidates}")
  return str(candidates[0]) if candidates else None


def _resolve_lora_path(exp_name, lora_root, ckpt_name):
  if lora_root is None:
    if exp_name is None:
      raise ValueError("Must provide either --lora_root or --exp_name")
    lora_root = f"data/checkpoints/siggraph-asia/multi_texture/{exp_name}"

  if os.path.basename(os.path.normpath(lora_root)).startswith("checkpoint-ep"):
    return lora_root

  if ckpt_name == "latest":
    candidates = sorted(glob.glob(os.path.join(lora_root, "checkpoint-ep*")))
    if not candidates:
      raise ValueError(f"No checkpoint directories found under: {lora_root}")
    return candidates[-1]

  return os.path.join(lora_root, ckpt_name)


def _clamp_window_bounds(xs_min, xs_max, ys_min, ys_max, control_w, control_h, res):
  xs_min = max(0, xs_min)
  ys_min = max(0, ys_min)
  xs_max = min(control_w, xs_max)
  ys_max = min(control_h, ys_max)

  if xs_max - xs_min < res:
    center_x = (xs_min + xs_max) // 2
    xs_min = max(0, min(control_w - res, center_x - res // 2))
    xs_max = xs_min + res

  if ys_max - ys_min < res:
    center_y = (ys_min + ys_max) // 2
    ys_min = max(0, min(control_h - res, center_y - res // 2))
    ys_max = ys_min + res

  return xs_min, xs_max, ys_min, ys_max


def _safe_path_component(value):
  sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
  return sanitized.strip("._") or "unnamed"


def _load_checkpoint_loras(pipe, lora_path):
  pipe.load_lora_weights(lora_path)

  controlnet_path = os.path.join(lora_path, CONTROLNET_LORA_WEIGHTS_NAME)
  controlnet_state_dict = None
  if os.path.isfile(controlnet_path):
    controlnet_state_dict = torch.load(controlnet_path, map_location="cpu")
  else:
    try:
      state_dict, _, _ = pipe.lora_state_dict(lora_path, return_alphas=True, return_lora_metadata=True)
    except Exception:
      state_dict = None
    if state_dict:
      controlnet_state_dict = {
        f'{k.replace("controlnet.", "")}': v for k, v in state_dict.items() if k.startswith("controlnet.")
      }

  if not controlnet_state_dict:
    return

  transformer_peft = getattr(pipe.transformer, "peft_config", None) or {}
  controlnet_peft = getattr(pipe.controlnet, "peft_config", None) or {}
  adapter_name = next(iter(transformer_peft), None)

  if adapter_name is None:
    adapter_name = next(iter(controlnet_peft), None)

  if adapter_name is None:
    available = list(transformer_peft.keys())
    raise RuntimeError(
      f"Checkpoint at {lora_path} has ControlNet LoRA weights but no transformer adapter was loaded. "
      f"Available transformer adapters: {available}"
    )

  if adapter_name not in controlnet_peft:
    if adapter_name not in transformer_peft:
      raise RuntimeError(
        f"Checkpoint at {lora_path} has {CONTROLNET_LORA_WEIGHTS_NAME} but adapter '{adapter_name}' "
        "was not present on the transformer."
      )
    pipe.controlnet.add_adapter(transformer_peft[adapter_name], adapter_name=adapter_name)

  controlnet_state_dict = convert_unet_state_dict_to_peft(controlnet_state_dict)
  incompatible_keys = set_peft_model_state_dict(pipe.controlnet, controlnet_state_dict, adapter_name=adapter_name)
  if incompatible_keys is not None:
    unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
    if unexpected_keys:
      source = controlnet_path if os.path.isfile(controlnet_path) else f"{lora_path}/pytorch_lora_weights.safetensors"
      print(f"Warning: unexpected ControlNet LoRA keys from {source}: {unexpected_keys}")

data_name = args.data_name  # "texture_00", "rock_00"
exp_name = args.exp_name  # "wall01_colormatch_rotate270", "wall02_colormatch_rotate270"
test_path = _resolve_test_path(args.test_path, args.test_dir)
lora_path = _resolve_lora_path(exp_name, args.lora_root, args.ckpt_name)
mask_path = _resolve_mask_path(test_path)
label_map_path = _resolve_label_map_path(args.label_map_path, args.test_dir)
prompt_path = _resolve_prompt_path(args.prompt_path, args.test_dir)

if args.num_inference_steps is not None:
  num_inference_steps = args.num_inference_steps
else:
  if args.model_type == "black-forest-labs/FLUX.1-schnell":
    num_inference_steps = 4
  else:
    num_inference_steps = 28

with open(prompt_path, 'r') as f:
  lines = [line.strip() for line in f.readlines() if line.strip()]

base_prompt_format = lines[0]
combiner = lines[1]
fall_back_prompt = lines[-1]
prompt_fragments = lines[2:-1]

expected_fragments = args.num_textures - 1

if len(prompt_fragments) != expected_fragments:
  raise ValueError(
    f"Number of prompt fragments ({len(prompt_fragments)}) in {prompt_path} "
    f"does not match --num_textures ({expected_fragments})."
  )

seed = 0
res = args.res
rotate_270 = 'rotate270' in lora_path
np.random.seed(seed)
# Stride controls tile spacing at each denoising step. A smaller stride at early steps means
# more overlapping windows, which helps the model agree on large-scale structure before details
# are committed. 'linear' increases stride progressively so early steps are denser.
if args.stride_method == "constant":
  stride_list = np.array([res//2] * num_inference_steps)
elif args.stride_method == "linear":
  stride_list = np.round(np.linspace(res//2, res//8*7, num_inference_steps)/16).astype(int)*16
elif args.stride_method == "random":
  assert False, "Have not made it divisible by 16 yet"
  stride_list = np.array([res//2] * num_inference_steps)
  stride_list[1:] = np.random.randint(res//2+1, res//8*7, size=num_inference_steps-1)
else:
  raise ValueError(f"Invalid stride method: {args.stride_method}")


# 2. load scale
scale_path = args.scale_path
scale = float(open(scale_path, 'r').readlines()[0].strip())
scale = 4.0 if args.ablation_name == 'ablation0' else scale


# 5. out paths
if args.datetime_str is None:
  datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
else:
  datetime_str = args.datetime_str

out_dir = os.path.join(lora_path, f"results_{datetime_str}_stride{args.stride_method}")
if args.debug:
  out_dir = out_dir + f"_debug{args.debug_size}"
if args.debug_oom:
  out_dir = out_dir + f"_debug_oom"
if args.use_pretrained:
  out_dir = out_dir + f"_pretrained"
if args.ablation_name is not None:
  out_dir = out_dir + f"_{args.ablation_name}"
if args.delete_prev:
  os.system(f"rm -rf {out_dir}")
sample_out_dir = os.path.join(out_dir, _safe_path_component(data_name))
os.makedirs(sample_out_dir, exist_ok=True)

out_path = os.path.join(sample_out_dir, f"md_decodemd_ds{scale}_stride{stride_list[0]}_nsteps{num_inference_steps}.png")
intermediate_dir = out_path.replace(".png", "_intermediate")
os.makedirs(intermediate_dir, exist_ok=True) 


# 3. get valid bounds
if mask_path is None:
  full_mask = Image.new("L", Image.open(test_path).size, 255)
else:
  full_mask = Image.open(mask_path).convert("L")
if rotate_270:
  full_mask = full_mask.rotate(90, expand=True)  # rotate 90 degrees to get horizontal version
full_mask_np = np.array(full_mask)
full_w, full_h = full_mask.size

# pad so that the image is divisible by 16
full_upscaled_w, full_upscaled_h = int(round(full_w*scale)), int(round(full_h*scale))
full_upscaled_w_pad = int(np.ceil(full_upscaled_w/16) * 16) - full_upscaled_w
full_upscaled_h_pad = int(np.ceil(full_upscaled_h/16) * 16) - full_upscaled_h

# extra padding in all directions
top_pad, bottom_pad, left_pad, right_pad = res, res, res, res

# make sure the crop coordinates are divisible by 16
full_ys, full_xs = np.where(full_mask_np == 255)
xs_min, xs_max = int(np.floor(full_xs.min() * scale / 16) * 16), int(np.ceil((full_xs.max()+1) * scale / 16) * 16)
ys_min, ys_max = int(np.floor(full_ys.min() * scale / 16) * 16), int(np.ceil((full_ys.max()+1) * scale / 16) * 16)

# shift all coords by padding
xs_min += left_pad
xs_max += left_pad
ys_min += top_pad
ys_max += top_pad

# get debug crop, centered around crop center
if args.debug:
  assert args.debug_size % 16 == 0, f"debug_size must be divisible by 16, got {args.debug_size}"
  xs_min = (xs_min + xs_max)//2
  xs_max = xs_min + args.debug_size
  ys_min = (ys_min + ys_max)//2
  ys_max = ys_min + args.debug_size

# extra padding in all directions
xs_min -= res
xs_max += res
ys_min -= res
ys_max += res

control_w_bound = full_upscaled_w + left_pad + right_pad + full_upscaled_w_pad
control_h_bound = full_upscaled_h + top_pad + bottom_pad + full_upscaled_h_pad
xs_min, xs_max, ys_min, ys_max = _clamp_window_bounds(
  xs_min, xs_max, ys_min, ys_max, control_w_bound, control_h_bound, res
)

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

if not args.remask:
  # Build per-timestep tile coordinates. label=0 marks border tiles (first/last two rows/cols)
  # that fall partly outside the object region; label=1 marks interior tiles whose output is
  # blended into the final result. Border tiles are still computed to give interior tiles
  # sufficient context, but their outputs are discarded during compositing.
  coords_list = []  # (h_starts, w_starts) for all time steps
  label_list = []  # 0 for border/padding tiles, 1 for valid interior tiles
  num_windows = []
  for i in range(num_inference_steps):

    stride = stride_list[i]
    h_starts = list(range(ys_min, ys_max-res+1, stride))
    if h_starts[-1] + res < ys_max:
        h_starts.append(ys_max - res)
    w_starts = list(range(xs_min, xs_max-res+1, stride))
    if w_starts[-1] + res < xs_max:
        w_starts.append(xs_max - res)

    coords_list_t = []
    label_list_t = []
    for i, h_start in enumerate(h_starts):
      for j, w_start in enumerate(w_starts):
        coords_list_t.append((h_start, w_start))
        label_list_t.append(0 if i <= 1 or j <= 1 or i >= len(h_starts)-2 or j >= len(w_starts)-2 else 1)
    coords_list.append(np.array(coords_list_t))
    label_list.append(np.array(label_list_t))
    num_windows.append(len(h_starts) * len(w_starts))

  print(f"Average num_windows per time step: {np.mean(num_windows):.2f}")
  print(f"Total num_windows: {np.sum(num_windows)}")
  print(f"Total estimated run time: {np.sum(num_windows)*18/num_inference_steps/60/60:.2f} hours")

  # HUY
  valid_coords_step0 = coords_list[0][label_list[0] == 1]
  target_coords_for_viz = []
  num_to_select = min(3, len(valid_coords_step0))
  if num_to_select > 0:
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(len(valid_coords_step0), num_to_select, replace=False)
    target_patches = valid_coords_step0[indices]

    h_win, w_win = args.res, args.res
    anchor_points_for_viz = [(h + h_win // 2, w + w_win // 2) for h, w in target_patches]

  # 6. preview inference regions
  inference_vis = np.zeros_like(full_mask_np)
  for (h, w), label in zip(coords_list[0], label_list[0]):
    if label == 1:
      lr_w, lr_h = int(round((w-res)/scale)), int(round((h-res)/scale))
      lr_res = int(round(res/scale))
      inference_vis[lr_h:lr_h+lr_res, lr_w:lr_w+lr_res] = 1
  inference_vis_pil = Image.fromarray((inference_vis*255).astype('uint8'))
  if rotate_270:
    inference_vis_pil = inference_vis_pil.rotate(270, expand=True)
  inference_vis_pil.save(os.path.join(out_dir, "inference_region_preview.png"))


  # 7a. upscale control image
  start_time = time.time()
  full_im = Image.open(test_path)
  if rotate_270:
    full_im = full_im.rotate(90, expand=True)
  control_im = full_im.resize((full_upscaled_w, full_upscaled_h), resample=Image.BICUBIC)
  control_im = add_margin(control_im, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, 0)
  control_w, control_h = control_im.size
  print(f"took {(time.time() - start_time):.2f} seconds to upscale control image from {full_w}x{full_h} to {control_w}x{control_h}")
  print('after upscale control image', psutil.Process().memory_info().rss / 1e9, "GB")
  control_im_copy = control_im.copy()

  # 7b. upscale label map
  label_map = Image.open(label_map_path).convert("L")
  if rotate_270:
    label_map = label_map.rotate(90, expand=True)
  label_map = label_map.resize((full_upscaled_w, full_upscaled_h), resample=Image.NEAREST)
  label_map = add_margin(label_map, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, 0)


  # 8. load pipeline
  start_time = time.time()
  controlnet = FluxControlNetModel.from_pretrained(
    "jasperai/Flux.1-dev-Controlnet-Upscaler",
    torch_dtype=torch.bfloat16
  )
  pipe = FluxControlNetMultiDiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    controlnet=controlnet,
    num_textures=args.num_textures,
    torch_dtype=torch.bfloat16
  ).to("cuda")
  # scheduler is FlowMatchEulerDiscreteScheduler

  if not args.use_pretrained:
    _load_checkpoint_loras(pipe, lora_path)
  print(f"Time taken to load model: {time.time() - start_time} seconds")
  print('after loading model', psutil.Process().memory_info().rss / 1e9, "GB")  


  # 9. inference
  with torch.no_grad():
    image, intermediate_list = pipe(
        base_prompt_format=base_prompt_format,
        combiner=combiner,
        fall_back_prompt=fall_back_prompt,
        prompt_fragments=prompt_fragments,
        control_image=control_im,
        label_map=label_map,
        num_textures=args.num_textures,
        coords_list=coords_list,
        label_list=label_list,
        controlnet_conditioning_scale=0.6,
        num_inference_steps=num_inference_steps, 
        guidance_scale=3.5,
        height=control_h,
        width=control_w,
        h_win=res,
        w_win=res,
        generator=torch.Generator("cpu").manual_seed(seed),
        out_dir=intermediate_dir,
        return_dict=False,
        use_gaussian_mask=args.use_gaussian_mask,
        sigma=args.sigma,
        debug_oom=args.debug_oom,
        debug_size=args.debug_size,
        chunk_size=args.chunk_size,
        anchor_points_for_viz= None # anchor_points_for_viz,
    )

else:
  start_time = time.time()
  image = Image.open(out_path)
  if rotate_270:
    image = image.rotate(90, expand=True)
  image = add_margin(image, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, (255, 255, 255))
  print(f"output image loaded in {(time.time() - start_time):.2f} seconds")
  import pdb; pdb.set_trace()

  # 7. upscale control image
  start_time = time.time()
  full_im = Image.open(test_path)
  if rotate_270:
    full_im = full_im.rotate(90, expand=True)
  control_im = full_im.resize((full_upscaled_w, full_upscaled_h), resample=Image.BICUBIC)
  control_im_copy = add_margin(control_im, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, (255, 255, 255))
  print(f"control image loaded in {(time.time() - start_time):.2f} seconds")
  import pdb; pdb.set_trace()

# # 10. save individual windows
# start_time = time.time()
# for i, im in enumerate(intermediate_list):
#   y, x = coords_list[0][i]
#   # im.rotate(270, expand=True).save(os.path.join(intermediate_dir, f"crop_y{y}_x{x}_score{scores[i]:.4f}.png"))
#   im.save(os.path.join(intermediate_dir, f"crop_y{y}_x{x}.png"))
# print(f"took {(time.time() - start_time):.2f} seconds to save the intermediate images")


# 11. upscale mask, convert to binary, mask out pixels, crop out padding
start_time = time.time()
control_mask = full_mask.resize((full_upscaled_w, full_upscaled_h), resample=Image.BICUBIC)
control_mask = Image.fromarray((np.array(control_mask) >= 128).astype(np.uint8) * 255)
control_mask = add_margin(control_mask, top_pad, right_pad+full_upscaled_w_pad, bottom_pad+full_upscaled_h_pad, left_pad, 0)
control_mask = control_mask.convert('L').convert('1')
image = Image.composite(image, control_im_copy, control_mask)

# crop padding
image = image.crop((left_pad, top_pad, full_upscaled_w+left_pad, full_upscaled_h+top_pad))
print(f"took {(time.time() - start_time):.2f} seconds to apply mask and unpad")


# 12. save the entire image
start_time = time.time()
if rotate_270:
  image = image.rotate(270, expand=True)
image.save(out_path)
print(f"took {(time.time() - start_time):.2f} seconds to save the output image, size: {image.size}")
start_time = time.time()
if rotate_270:
  # already rotated, h should be w
  image.resize((full_h, full_w), resample=Image.BICUBIC).save(out_path.replace(".png", "_preview.png"))
else:
  image.resize((full_w, full_h), resample=Image.BICUBIC).save(out_path.replace(".png", "_preview.png"))

print(f"took {(time.time() - start_time):.2f} seconds to resize and save the preview image")


def _stage_result(source_path, staged_path, mode):
  if staged_path is None:
    return
  staged = Path(staged_path)
  staged.parent.mkdir(parents=True, exist_ok=True)
  if staged.exists() or staged.is_symlink():
    staged.unlink()
  if mode == "copy":
    shutil.copy2(source_path, staged)
  else:
    staged.symlink_to(Path(source_path).resolve())


_stage_result(out_path, args.stage_result_path, args.stage_mode)


# 13. create the dzi file and tiles (requires the `deepzoom` package: pip install deepzoom)
if args.write_dzi:
  dzi_dir = intermediate_dir.replace("_intermediate", "_dzi")
  os.makedirs(dzi_dir, exist_ok=True)

  full_im_path = os.path.join(dzi_dir, "full_im.png")
  full_im.save(full_im_path)

  html_path = os.path.join(dzi_dir, "index_compare.html")
  with open(html_path, "w") as f:
    f.write(f"""
<!DOCTYPE html>
<html>
  <head>
      <title>{data_name}</title>
      <script src="openseadragon/openseadragon.min.js"></script>
      <style>
          .viewer-container {{
              display: flex;
              width: 100%;
              height: 100vh;
          }}
          .viewer {{
              width: 50%;
              height: 100%;
          }}
          img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
          }}
      </style>
  </head>
  <body>
      <div class="viewer-container">
          <div id="viewer1" class="viewer"></div>
          <div id="viewer2" class="viewer"></div>
      </div>
      
      <script type='text/javascript'>
          // Initialize both viewers
          var viewer1 = OpenSeadragon({{
              id: "viewer1",
              prefixUrl: "openseadragon/images/",
              tileSources: {{
                type: 'image',
                url: './full_im.png'
              }},
              showNavigationControl: true,
              defaultZoomLevel: 1,
              minZoomLevel: 0.5,
              maxZoomLevel: {scale:.3f}
          }});
          var viewer2 = OpenSeadragon({{
              id: "viewer2",
              prefixUrl: "openseadragon/images/",
              tileSources: './image.dzi',
              maxZoomPixelRatio: 1,
              showNavigationControl: true
          }});
          viewer1.setControlsEnabled(false);
          viewer2.setControlsEnabled(false);

          var label1 = document.createElement('div');
          label1.innerHTML = "Closeup (captured, {scale:.3f}x)";
          label1.style.position = "absolute";
          label1.style.top = "10px";
          label1.style.right = "10px";
          label1.style.background = "green";
          label1.style.color = "white";
          label1.style.padding = "6px";
          label1.style.fontSize = "24px";
          label1.style.zIndex = 1000;
          viewer1.container.appendChild(label1);


          var label2 = document.createElement('div');
          label2.innerHTML = "Ours";
          label2.style.position = "absolute";
          label2.style.top = "10px";
          label2.style.right = "10px";
          label2.style.background = "green";
          label2.style.color = "white";
          label2.style.padding = "6px";
          label2.style.fontSize = "24px";
          label2.style.zIndex = 1000;
          viewer2.container.appendChild(label2);

          var zoomLabel = document.createElement("div");
          zoomLabel.style.position = "absolute";
          zoomLabel.style.top = "60px";
          zoomLabel.style.right = "10px";
          zoomLabel.style.background = "green";
          zoomLabel.style.color = "white";
          zoomLabel.style.padding = "6px";
          zoomLabel.style.fontSize = "24px";
          zoomLabel.style.zIndex = 1000;
          viewer2.container.appendChild(zoomLabel);

          function updateZoomLabel() {{

            const MAX_SCALE = {scale:.3f};
            const viewportZoom = viewer2.viewport.getZoom(true);
            const homeZoom = viewer2.viewport.getHomeZoom();

            const maxZoom = viewer2.viewport.getMaxZoom();

            const scaleFactor = 1 + (viewportZoom - homeZoom) * (MAX_SCALE - 1) / (maxZoom - homeZoom);

            const displayScale = Math.max(1, scaleFactor);

            zoomLabel.innerHTML = `Zoom: ${{displayScale.toFixed(3)}}x`;
          }}

          viewer2.addHandler("open", updateZoomLabel);
          viewer2.addHandler("zoom", updateZoomLabel);
          viewer2.addHandler("animation", updateZoomLabel);

      </script>
      
  </body>
</html>
  """)
  # Copy the bundled OpenSeadragon viewer into the output directory
  _src_dir = os.path.dirname(os.path.abspath(__file__))
  osd_src = os.path.join(_src_dir, 'openseadragon')
  osd_dst = os.path.join(dzi_dir, 'openseadragon')
  if os.path.exists(osd_dst):
    shutil.rmtree(osd_dst)
  shutil.copytree(osd_src, osd_dst)
  # Create DZI tiles; the HTML viewer expects ./image.dzi relative to dzi_dir
  creator = deepzoom.ImageCreator(tile_size=254, tile_overlap=1, tile_format="jpg")
  creator.create(out_path, os.path.join(dzi_dir, "image.dzi"))


  html_path = os.path.join(dzi_dir, "index.html")
  with open(html_path, "w") as f:
    f.write(f"""
<!DOCTYPE html>
<html>
  <head>
    <title>{data_name}</title>
    <script src="openseadragon/openseadragon.min.js"></script>
    <style>
      body {{
        margin: 0;
      }}
      .viewer-container {{
        display: flex;
        width: 100%;
        height: 100vh;
      }}
      .viewer {{
        width: 100%;
        height: 100%;
      }}
    </style>
  </head>
  <body>
    <div class="viewer-container">
      <div id="viewer2" class="viewer"></div>
    </div>

    <script type="text/javascript">
      var viewer2 = OpenSeadragon({{
        id: "viewer2",
        prefixUrl: "openseadragon/images/",
        tileSources: "./image.dzi",
        showNavigationControl: true,
        showNavigator: true,
        navigatorPosition: "BOTTOM_RIGHT",
        navigatorSizeRatio: 0.2
      }});

      viewer2.setControlsEnabled(false);

      var label2 = document.createElement("div");
      label2.innerHTML = "Our Result ({scale:.3f}x)";
      label2.style.position = "absolute";
      label2.style.top = "10px";
      label2.style.right = "10px";
      label2.style.background = "green";
      label2.style.color = "white";
      label2.style.padding = "6px";
      label2.style.fontSize = "24px";
      label2.style.zIndex = 1000;
      viewer2.container.appendChild(label2);

      var zoomLabel = document.createElement("div");
      zoomLabel.style.position = "absolute";
      zoomLabel.style.top = "60px";
      zoomLabel.style.right = "10px";
      zoomLabel.style.background = "green";
      zoomLabel.style.color = "white";
      zoomLabel.style.padding = "6px";
      zoomLabel.style.fontSize = "24px";
      zoomLabel.style.zIndex = 1000;
      viewer2.container.appendChild(zoomLabel);

      function updateZoomLabel() {{

        const MAX_SCALE = {scale:.3f};
        const viewportZoom = viewer2.viewport.getZoom(true);
        const homeZoom = viewer2.viewport.getHomeZoom();

        const maxZoom = viewer2.viewport.getMaxZoom();

        const scaleFactor = 1 + (viewportZoom - homeZoom) * (MAX_SCALE - 1) / (maxZoom - homeZoom);

        const displayScale = Math.max(1, scaleFactor);

        zoomLabel.innerHTML = `Zoom: ${{displayScale.toFixed(3)}}x`;
      }}

      viewer2.addHandler("open", updateZoomLabel);
      viewer2.addHandler("zoom", updateZoomLabel);
      viewer2.addHandler("animation", updateZoomLabel);
      
    </script>
  </body>
</html>
  """)

print("--------------------------------")
print(f"Total time taken: {(time.time() - global_start_time):.2f} seconds")
