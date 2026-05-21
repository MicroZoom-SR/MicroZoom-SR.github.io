# MicroZoom

This repository contains the official implementation of **MicroZoom**, a system for generating gigapixel-scale images from close-up macro photographs and a single full-view reference image. MicroZoom uses per-instance fine-tuning of a Flux.1-dev ControlNet with LoRA adapters, combined with multi-texture label-map conditioning and a sliding-window multi-diffusion inference pipeline.

---

## System Requirements

| Task | GPU |
|------|-----|
| Training | NVIDIA A100 (80 GB) or equivalent |
| Inference | NVIDIA A40 (40 GB) or equivalent |

---

## Installation

```bash
conda create -n microzoom python=3.10
conda activate microzoom

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install diffusers from source (requires 0.33.0.dev0+)
pip install git+https://github.com/huggingface/diffusers.git

# Install remaining dependencies
pip install -r requirements.txt
```

You will also need access to the base models from Hugging Face:
- [`black-forest-labs/FLUX.1-dev`](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [`jasperai/Flux.1-dev-Controlnet-Upscaler`](https://huggingface.co/jasperai/Flux.1-dev-Controlnet-Upscaler)

Authenticate with `hf auth login` before downloading gated models.

---

## Sample Data

The `sample_data/` directory contains two example objects (`tote_cascade1`, `tote_cascade2`), each with the following structure:

```
sample_data/<object>/
├── <name>_close00.jpg, ...      # close-up macro photographs
├── <name>_close00_label_map.png, ...  # per-image texture label maps
├── <name>_full.jpg              # full-view reference image
├── <name>_full_label_map.png    # full-view texture label map
├── <name>_full_mask.png         # binary object mask
├── prompts.txt                  # per-image text prompts for training
├── inference_prompts.txt        # prompt template for inference
└── registration/
    ├── closeup00_final_mask.png, ...  # registration masks per close-up
    └── scale.txt                # upscaling factor (closeup / full resolution)
```

---

## Training

Train a per-instance LoRA on the sample tote bag object:

```bash
accelerate launch src/train.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-dev \
  --instance_data_dir "sample_data/tote_cascade1/tote_close.jpg" \
  --closeup_label_map_root "sample_data/tote_cascade1/tote_close.png" \
  --val_data_dir sample_data/tote_cascade1/tote_full.jpg \
  --full_label_map_root sample_data/tote_cascade1/tote_full_label_map.png \
  --instance_prompt "detailed close-up photo of tote bag textures" \
  --validation_prompt "detailed close-up photo of tote bag textures" \
  --num_textures 4 \
  --output_dir outputs/tote_cascade1 \
  --test_vis_dir outputs/tote_cascade1/vis \
  --resolution 1024 \
  --mixed_precision bf16 \
  --gradient_accumulation_steps 4 \
  --steps_per_epoch 250 \
  --num_train_epochs 2 \
  --learning_rate 1e-4 \
  --rank 4 \
  --checkpointing_epochs 1
```

Key arguments:

| Argument | Description |
|----------|-------------|
| `--instance_data_dir` | Glob pattern for close-up training images |
| `--closeup_label_map_root` | Glob pattern for per-image texture label maps |
| `--val_data_dir` | Path to the full-view reference image |
| `--full_label_map_root` | Label map for the full-view image (used for validation) |
| `--num_textures` | Number of texture classes in the label maps |
| `--resolution` | Training crop size in pixels |
| `--rank` | LoRA rank for both the transformer and ControlNet adapters |
| `--steps_per_epoch` | Number of gradient steps per epoch |
| `--num_train_epochs` | Total epochs; the first half uses label-map conditioning only, the second half adds ControlNet guidance |

Checkpoints are saved to `--output_dir/checkpoint-ep{N:04d}/`.

---

## Inference

Run multi-diffusion inference on the held-out close-up from `tote_cascade1`:

```bash
python src/inference.py \
  --data_name tote_cascade1 \
  --test_dir sample_data/tote_cascade1 \
  --lora_root outputs/tote_cascade1/checkpoint-ep0002 \
  --scale_path outputs/tote_cascade1/vis/final_scale.txt \
  --prompt_path sample_data/tote_cascade1/inference_prompts.txt \
  --num_textures 4 \
  --res 1024 \
  --stride_method linear \
  --use_gaussian_mask
```

Add `--write_dzi` to generate a local OpenSeadragon deep-zoom viewer alongside the output.

Key arguments:

| Argument | Description |
|----------|-------------|
| `--data_name` | Name used for the output subdirectory |
| `--test_dir` | Directory containing the held-out close-up and its label map |
| `--lora_root` | Path to a trained checkpoint directory |
| `--scale_path` | Path to `registration/scale.txt` with the upscaling factor |
| `--res` | Tile resolution in pixels (should match training resolution) |
| `--stride_method` | Tile stride schedule: `constant`, `linear` (recommended), or `random` |
| `--use_gaussian_mask` | Blend tile boundaries with a Gaussian weight mask |
| `--write_dzi` | Save a Deep Zoom Image viewer for interactive exploration |
| `--num_inference_steps` | Override the default number of denoising steps (28 for FLUX.1-dev) |

Results are saved under `<lora_root>/results_<datetime>_stride<method>/<data_name>/`.