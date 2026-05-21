#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

"""
    transformer has 11,901,408,320 parameters, rank-4 lora has 22,413,312 parameters
    controlnet has 1,791,605,248 parameters, rank-4 lora has 4,423,680 parameters
"""

import cv2
import argparse
import copy
import inspect
import itertools
import logging
import math
import os
import random
import shutil
import warnings
from contextlib import nullcontext
from pathlib import Path
import glob
import mediapy
import imageio

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageFilter, ExifTags
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # FluxPipeline,
    FluxTransformer2DModel,
    FluxControlNetModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from diffusers_custom.pipelines.flux.pipeline_flux import FluxPipeline
from microzoom.pipelines.flux.custom_flux_controlnet_flux_control_v3 import CustomFluxControlNetPipeline

# lpips loss
import lpips
def pil_to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor(),                   # Convert to tensor in range [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5),   # Normalize to range [-1, 1]
                             (0.5, 0.5, 0.5))
    ])
    tensor = transform(img).unsqueeze(0)        # Add batch dimension (1, 3, H, W)
    return tensor


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__)
CONTROLNET_LORA_WEIGHTS_NAME = "controlnet_lora_layers.pt"


def save_model_card(
    repo_id: str,
    images=None,
    base_model: str = None,
    train_text_encoder=False,
    instance_prompt=None,
    validation_prompt=None,
    repo_folder=None,
):
    widget_dict = []
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            widget_dict.append(
                {"text": validation_prompt if validation_prompt else " ", "output": {"url": f"image_{i}.png"}}
            )

    model_description = f"""
# Flux DreamBooth LoRA - {repo_id}

<Gallery />

## Model description

These are {repo_id} DreamBooth LoRA weights for {base_model}.

The weights were trained using [DreamBooth](https://dreambooth.github.io/) with the [Flux diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md).

Was LoRA for the text encoder enabled? {train_text_encoder}.

## Trigger words

You should use `{instance_prompt}` to trigger the image generation.

## Download model

[Download the *.safetensors LoRA]({repo_id}/tree/main) in the Files & versions tab.

## Use it with the [🧨 diffusers library](https://github.com/huggingface/diffusers)

```py
from diffusers import AutoPipelineForText2Image
import torch
pipeline = AutoPipelineForText2Image.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to('cuda')
pipeline.load_lora_weights('{repo_id}', weight_name='pytorch_lora_weights.safetensors')
image = pipeline('{validation_prompt if validation_prompt else instance_prompt}').images[0]
```

For more details, including weighting, merging and fusing LoRAs, check the [documentation on loading LoRAs in diffusers](https://huggingface.co/docs/diffusers/main/en/using-diffusers/loading_adapters)

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=model_description,
        widget=widget_dict,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "lora",
        "flux",
        "flux-diffusers",
        "template:sd-lora",
    ]

    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


@torch.no_grad()
def log_validation(
    pipeline,
    args,
    accelerator,
    pipeline_args,
    epoch,
    torch_dtype,
    groundtruth,
    control_image,
    label_maps_val,
    groundtruth_train,
    control_image_train,
    label_maps_train,
    lr_size, 
    hr_size,
    is_final_validation=False,
    val_real_lr_crops = None,
    val_real_hr_crops = None,
    train_prompts=None,
    stage1=None,
):
    is_real = '2025' in args.val_data_dir

    # converts tensor to image
    def tensor_to_pil(tensor):
        tensor = (tensor + 1) / 2  # Scale from (-1, 1) to (0, 1)
        tensor = tensor.clamp(0, 1) * 255 # Clamp to ensure valid range
        return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8))

    logger.info(
        f"Running validation... \n Generating {len(control_image)+len(control_image_train)} images with prompt:"
        f" {args.validation_prompt}."
    )

    # progress bar
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # load accelerator
    autocast_ctx = torch.autocast(accelerator.device.type) if not is_final_validation else nullcontext()

    # generate train/validation images
    with autocast_ctx:
        train_images = []
        for i, control_image_ in enumerate(control_image_train):
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None

            current_pipeline_args = pipeline_args.copy()
            current_pipeline_args["prompt"] = train_prompts[i]

            if type(control_image_) == torch.Tensor:
                image = pipeline(
                    **current_pipeline_args, 
                    generator=generator, 
                    control_image=control_image_.unsqueeze(0).to(torch.float16), 
                    label_map=label_maps_train[i].unsqueeze(0).to(torch.float16),
                    num_textures=args.num_textures, 
                    stage1=stage1,
                ).images[0]
            else:
                image = pipeline(
                    **current_pipeline_args, 
                    generator=generator, 
                    control_image=control_image_, 
                    label_map=label_maps_train[i], 
                    num_textures=args.num_textures,
                    stage1=stage1
                ).images[0]
            train_images.append(image)
            print(f"{i+1} out of {len(control_image_train)} train images generated.")

        val_images = []
        for i, control_image_ in enumerate(control_image):
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
            if type(control_image_) == torch.Tensor:
                image = pipeline(
                    **pipeline_args, 
                    generator=generator, 
                    control_image=control_image_.unsqueeze(0).to(torch.float16), 
                    label_map=label_maps_val[i].unsqueeze(0).to(torch.float16),
                    num_textures=args.num_textures, 
                    stage1=stage1
                ).images[0]
            else:
                image = pipeline(
                    **pipeline_args, 
                    generator=generator, 
                    control_image=control_image_, 
                    label_map = label_maps_val[i], 
                    num_textures=args.num_textures,
                    stage1=stage1
                ).images[0]
            val_images.append(image)
            print(f"{i+1} out of {len(control_image)} val images generated.")

    if type(groundtruth_train[0]) == torch.Tensor:
        train_loss_l2 = [((np.array(image).astype(float)/255*2-1) - np.transpose(image_gt.numpy(), (1,2,0)))**2 for image, image_gt in zip(train_images, groundtruth_train) if image_gt is not None]
    else:
        train_loss_l2 = [((np.array(image).astype(float)/255*2-1) - (np.array(image_gt).astype(float)/255*2-1))**2 for image, image_gt in zip(train_images, groundtruth_train) if image_gt is not None]
    train_loss_l2 = np.mean(np.stack(train_loss_l2, axis=0))

    train_images_lr = [image.resize((lr_size, lr_size), Image.Resampling.BICUBIC).resize((hr_size, hr_size), Image.Resampling.BICUBIC) for image in train_images]
    if type(control_image_train[0]) == torch.Tensor:
        train_lr_loss_l2 = [((np.array(image).astype(float)/255*2-1) - np.transpose(control_image_train[i].numpy(), (1,2,0)))**2 for i, image in enumerate(train_images_lr)]
    else:
        train_lr_loss_l2 = [((np.array(image).astype(float)/255*2-1) - (np.array(control_image_train[i]).astype(float)/255*2-1))**2 for i, image in enumerate(train_images_lr)]
    train_lr_loss_l2 = np.mean(np.stack(train_lr_loss_l2, axis=0))

    
    if is_real:
        val_loss_l2 = 0
    else:
        if groundtruth[0] is None:
            val_loss_l2 = 0
        elif type(groundtruth[0]) == torch.Tensor:
            val_loss_l2 = [((np.array(image).astype(float)/255*2-1) - np.transpose(image_gt.numpy(), (1,2,0)))**2 for image, image_gt in zip(val_images, groundtruth) if image_gt is not None]
            val_loss_l2 = np.mean(np.stack(val_loss_l2, axis=0))
        else:
            val_loss_l2 = [((np.array(image).astype(float)/255*2-1) - (np.array(image_gt).astype(float)/255*2-1))**2 for image, image_gt in zip(val_images, groundtruth) if image_gt is not None]
            val_loss_l2 = np.mean(np.stack(val_loss_l2, axis=0))
        # import pdb; pdb.set_trace()
    
    val_images_lr = [image.resize((lr_size, lr_size), Image.Resampling.BICUBIC).resize((hr_size, hr_size), Image.Resampling.BICUBIC) for image in val_images]
    if type(control_image[0]) == torch.Tensor:
        val_lr_loss_l2 = [((np.array(image).astype(float)/255*2-1) - np.transpose(control_image[i].numpy(), (1,2,0)))**2 for i, image in enumerate(val_images_lr)]
    else:
        val_lr_loss_l2 = [((np.array(image).astype(float)/255*2-1) - (np.array(control_image[i]).astype(float)/255*2-1))**2 for i, image in enumerate(val_images_lr)]
    val_lr_loss_l2 = np.mean(np.stack(val_lr_loss_l2, axis=0))

    # lpips loss
    loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
    image_tensor, gt_tensor, train_loss_lpips = [], [], []
    for image, image_gt in zip(train_images, groundtruth_train):
        if image_gt is not None:
            image_tensor.append(pil_to_tensor(image))
            if type(image_gt) == torch.Tensor:
                train_loss_lpips.append(loss_fn_alex(image_tensor[-1], image_gt.unsqueeze(0)).item())
            else:
                train_loss_lpips.append(loss_fn_alex(image_tensor[-1], pil_to_tensor(image_gt)).item())
    train_loss_lpips = np.mean(train_loss_lpips)

    if is_real:
        val_loss_lpips = 0
    else:
        if groundtruth[0] is None:
            val_loss_lpips = 0
        else:
            image_tensor, gt_tensor, val_loss_lpips = [], [], []
            for image, image_gt in zip(val_images, groundtruth):
                if image_gt is not None:
                    image_tensor.append(pil_to_tensor(image))
                    if type(image_gt) == torch.Tensor:
                        val_loss_lpips.append(loss_fn_alex(image_tensor[-1], image_gt.unsqueeze(0)).item())
                    else:
                        val_loss_lpips.append(loss_fn_alex(image_tensor[-1], pil_to_tensor(image_gt)).item())
            val_loss_lpips = np.mean(val_loss_lpips)
    
    # HUY VALIDATION LOGIC
    val_real_images, val_real_loss_l2, val_real_loss_lpips = [], None, None
    if val_real_lr_crops is not None and len(val_real_lr_crops) > 0:
        logger.info(f"Running real validation... \n Generating {len(val_real_lr_crops)} images.")
        with autocast_ctx:
            for i, control_image_ in enumerate(val_real_lr_crops):
                generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed is not None else None
                image = pipeline(**pipeline_args, generator=generator, control_image=control_image_.unsqueeze(0).to(dtype=torch_dtype)).images[0]
                val_real_images.append(image)
            
            val_real_loss_l2 = [((np.array(image).astype(float)/255*2-1) - np.transpose(image_hr.numpy(), (1, 2, 0)))**2 for image, image_hr in zip(val_real_images, val_real_hr_crops)]
            val_real_loss_l2 = np.mean(np.stack(val_real_loss_l2, axis=0))

            real_val_loss_lpips_list = []
            for image, image_hr in zip(val_real_images, val_real_hr_crops):
                image_tensor = pil_to_tensor(image)
                real_val_loss_lpips_list.append(loss_fn_alex(image_tensor, image_hr.unsqueeze(0).to(image_tensor.device)).item())
            val_real_loss_lpips = np.mean(real_val_loss_lpips_list)

    # convert images for wandb
    control_image_vis = [tensor_to_pil(image) for image in control_image] if isinstance(control_image[0], torch.Tensor) else control_image
    groundtruth_vis = []
    for i, image in enumerate(groundtruth):
        if image is None:
            groundtruth_vis.append(Image.fromarray(np.ones((1024, 1024, 3), dtype=np.uint8) * 255))
        elif isinstance(image, torch.Tensor):
            groundtruth_vis.append(tensor_to_pil(image))
        else:
            groundtruth_vis.append(image)
    control_image_train_vis = [tensor_to_pil(image) for image in control_image_train] if isinstance(control_image_train[0], torch.Tensor) else control_image_train
    groundtruth_train_vis = [tensor_to_pil(image) for image in groundtruth_train] if isinstance(groundtruth_train[0], torch.Tensor) else groundtruth_train


    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            log_dict = {
                "val_loss_l2": val_loss_l2,
                "val_loss_l2_lowres": val_lr_loss_l2,
                "val_loss_lpips": val_loss_lpips,
                "train_loss_l2": train_loss_l2,
                "train_loss_l2_lowres": train_lr_loss_l2,
                "train_loss_lpips": train_loss_lpips,
                phase_name: [
                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(val_images)
                ],
                phase_name + "_ds": [
                    wandb.Image(image, caption=f"{i}") for i, image in enumerate(val_images_lr)
                ],
                phase_name + "_input_lr": [
                    wandb.Image(image, caption=f"{i}") for i, image in enumerate(control_image_vis)
                ],
                phase_name + "_groundtruth": [
                    wandb.Image(image, caption=f"{i}") for i, image in enumerate(groundtruth_vis) 
                ],
                "train": [
                    wandb.Image(image, caption=f"{i}: {train_prompts[i] if train_prompts else args.validation_prompt}") for i, image in enumerate(train_images)
                ],
                "train_ds": [
                    wandb.Image(image, caption=f"{i}") for i, image in enumerate(train_images_lr)
                ],
                "train_input_lr": [
                    wandb.Image(image, caption=f"{i}") for i, image in enumerate(control_image_train_vis)
                ],
                "train_groundtruth": [
                    wandb.Image(image, caption=f"{i}") for i, image in enumerate(groundtruth_train_vis) 
                ],
            }

            if val_real_images:
                val_real_hr_vis = [tensor_to_pil(img) for img in val_real_hr_crops]
                val_real_lr_vis = [tensor_to_pil(img) for img in val_real_lr_crops]
                log_dict.update({
                    "real_val_loss_l2": val_real_loss_l2,
                    "val_real_loss_lpips": val_real_loss_lpips,
                    "validation_real_output": [
                        wandb.Image(image, caption=f"{i}") for i, image in enumerate(val_real_images)
                    ],
                    "validation_real_lr_input": [
                        wandb.Image(image, caption=f"{i}") for i, image in enumerate(val_real_lr_vis)
                    ],
                    "validation_real_gt": [
                        wandb.Image(image, caption=f"{i}") for i, image in enumerate(val_real_hr_vis)
                    ]
                })
            
            tracker.log(log_dict)

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    free_memory()

    # save train/val input/output/groundtruth
    for i, image in enumerate(train_images):
        image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/train_output_{i}.png")
    for i, image in enumerate(val_images):
        image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/val_output_{i}.png")
    for i, image in enumerate(groundtruth_train_vis):
        image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/train_groundtruth_{i}.png")
    for i, image in enumerate(groundtruth_vis):
        image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/val_groundtruth_{i}.png")
    for i, image in enumerate(control_image_train_vis):
        image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/train_input_{i}.png")
    for i, image in enumerate(control_image_vis):
        image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/val_input_{i}.png")

    if val_real_images:
        val_real_hr_vis = [tensor_to_pil(img) for img in val_real_hr_crops]
        val_real_lr_vis = [tensor_to_pil(img) for img in val_real_lr_crops]
        for i, image in enumerate(val_real_images):
            image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/val_real_output_{i}.png")
        for i, image in enumerate(val_real_lr_vis):
            image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/val_real_input_lr_{i}.png")
        for i, image in enumerate(val_real_hr_vis):
            image.save(f"{args.output_dir}/checkpoint-ep{epoch:04d}/val_real_groundtruth_{i}.png")

    return train_images + val_images


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--real_val_path",
        type=str,
        default=None,
        help="Path to directory with paired low-res and high-res validation images",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) containing the training data of instance images (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help=("A folder containing the training data. "),
    )

    parser.add_argument(
        "--closeup_label_map_root",
        type=str,
        default=None,
        help=("A path containing label maps for the training data.")
    )

    parser.add_argument(
        "--full_label_map_root",
        type=str,
        default=None,
        help=("A path to a label map for the full image used for validation")
    )

    parser.add_argument(
        "--num_textures",
        type=int,
        default=3,
        help=("The number of textures for the label map")
    )

    parser.add_argument(
        "--val_data_dir",
        type=str,
        default=None,
        help=("A folder containing the validation data. "),
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help=("Zoom image, reference object mask"),
    )
    parser.add_argument(
        "--scale_path",
        type=str,
        default=None,
        help=("txt file with precomputed relative scales"),
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )

    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing the target image. By "
        "default, the standard Image Dataset maps out 'file_name' "
        "to 'image'.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default=None,
        help="The column of the dataset containing the instance prompt for each image",
    )

    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")

    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        required=False,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        required=True,
        help="The prompt with identifier specifying the instance, e.g. 'photo of a TOK dog', 'in the style of TOK'",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--test_vis_dir",
        type=str,
        default=None,
        help="The output directory to visualize test images",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--extra_scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--total_scale",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--add_blur",
        action="store_true",
    )
    parser.add_argument(
        "--blur_sigma",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_text_encoder",
        action="store_true",
        help="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=1000,
        help="Total number of training steps to perform per epoch.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpointing_epochs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=3.5,
        help="the FLUX.1 dev variant is a guidance distilled model",
    )

    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=0.6,
    )

    parser.add_argument(
        "--text_encoder_lr",
        type=float,
        default=5e-6,
        help="Text encoder learning rate to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--cache_latents",
        action="store_true",
        default=False,
        help="Cache the VAE latents",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_name",
        type=str,
        default="dreambooth-flux-dev-lora",
        help="Tracker project/name passed to accelerator.init_trackers.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Optional deterministic Weights & Biases run name override.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        default=False,
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--align_full_to_closeup", action="store_true", help="Align full image to closeup image")
    parser.add_argument("--skip_training", action="store_true", help="Skip training")
    parser.add_argument("--jpeg_degradation", action="store_true", help="Add jpeg degradation to the training data")
    parser.add_argument("--jpeg_quality", type=int, default=75, help="jpeg quality for jpeg degradation")
    parser.add_argument("--ablation_name", type=str, default=None)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.instance_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--instance_data_dir`")

    if args.dataset_name is not None and args.instance_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--instance_data_dir`")

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.class_data_dir is None:
            raise ValueError("You must specify a data directory for class images.")
        if args.class_prompt is None:
            raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args


def load_image_correct_orientation(path):
    image = Image.open(path)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation_value = exif.get(orientation, None)
            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"[WARN] EXIF orientation fix failed: {e}")
    return image

def tone_preserving_resize(img_pil, scale=4, blur_radius=1.0):
    """
    img_pil: PIL.Image in RGB, range [0, 255]
    Returns: PIL.Image resized back to original size, tone-preserving
    """

    blur_radius = 0.5 * scale
    
    # Convert to numpy float32, [0, 1]
    img = np.asarray(img_pil).astype(np.float32) / 255.0

    # Gamma expand to linear
    img_lin = np.clip(img, 1e-6, 1.0) ** 2.2

    # Convert back to PIL for blur
    img_lin_pil = Image.fromarray((img_lin * 255).astype(np.uint8))
    img_blur = img_lin_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Downsample
    w, h = img_blur.size
    new_size = (int(round(w/scale)), int(round(h/scale)))
    img_down = img_blur.resize(new_size, Image.BICUBIC)

    # Upsample back
    img_up = img_down.resize((w, h), Image.BICUBIC)

    # Convert to linear numpy
    img_up_np = np.asarray(img_up).astype(np.float32) / 255.0
    img_up_lin = np.clip(img_up_np, 0, 1)  # ensure valid range

    # Gamma compress
    img_out = np.clip(img_up_lin, 0, 1) ** (1 / 2.2)

    # Back to PIL
    return Image.fromarray((img_out * 255).round().astype(np.uint8)), blur_radius


def collate_fn(examples, with_prior_preservation=False):
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values_lr = [example["instance_images_lr"] for example in examples]

    pixel_values_label_map = [example["instance_label_map"] for example in examples]

    prompts = [example["instance_prompt"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        pixel_values += [example["class_images"] for example in examples]
        prompts += [example["class_prompt"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    pixel_values_lr = torch.stack(pixel_values_lr)
    pixel_values_lr = pixel_values_lr.to(memory_format=torch.contiguous_format).float()
    pixel_values_label_map = torch.stack(pixel_values_label_map)
    pixel_values_label_map = pixel_values_label_map.to(memory_format=torch.contiguous_format).long()

    batch = {
        "pixel_values": pixel_values, 
        "pixel_values_lr": pixel_values_lr, 
        "pixel_values_label_map": pixel_values_label_map, 
        "prompts": prompts
    }
    return batch

class OnlineUpscalerDreamBoothDataset(Dataset):
    """Dataset for per-instance fine-tuning of the MicroZoom upscaler.

    Expected directory layout (one object per training run):
        <instance_data_root>/
            <name>_close00.jpg, <name>_close01.jpg, ...  # close-up captures
            <name>_close00_label_map.png, ...             # integer label maps (0..num_textures-1)
            prompts.txt                                    # optional per-image text prompts
            registration/
                closeup00_final_mask.png, ...             # registration masks per close-up
                scale.txt                                  # upscaling factor relative to full image
        <val_data_root>          # full-view image (e.g. <name>_full.jpg)
        <val_data_root_mask>     # binary object mask (e.g. <name>_full_mask.png)
        <full_label_map_root>    # label map for the full image

    Each training step returns a random crop from a randomly selected close-up image,
    together with its synthetic low-resolution version (tone-preserving downscale) and the
    corresponding label map crop for multi-texture conditioning.
    """

    def __init__(
        self,
        args,
        instance_data_root,             # Path pattern to close-up images
        val_data_root,                  # Path to full image (used for validating model performance on non-training examples as a sanity check and manual checkpoint selection)
        test_vis_dir,                   # Directory to save visualizations and debug images
        instance_prompt,                # Text prompt to describe the image
        size=1024,                      # Size of image patches used for training and validation
        closeup_label_map_root=None,    # Path to training label maps
        full_label_map_root=None,       # Path to full image label map (used for model validation)
        real_val_path=None,             # Path to low res validation patches for real validation
    ):
        # set params
        self._length = args.steps_per_epoch  # steps per epoch
        self.size = size
        self.instance_prompt = instance_prompt
        self.custom_instance_prompts = None
        self.normal_control = args.normal_control = False
        os.makedirs(test_vis_dir, exist_ok=True)


        # initialize validation set
        self.val_real_lr_crops = []
        self.val_real_hr_crops = []

        # load training image paths and label map
        instance_data_root = sorted(glob.glob(instance_data_root.replace('.jpg', '*.jpg')))
        instance_data_root = [path for path in instance_data_root if not 'colormatched' in path]
        closeup_label_map_root = sorted(glob.glob(closeup_label_map_root.replace('.png', '*_label_map.png')))

        if not instance_data_root:
            raise ValueError(
                f"No close-up training images matched instance_data_root={args.instance_data_dir!r}"
            )
        if not closeup_label_map_root:
            raise ValueError(
                f"No close-up label maps matched closeup_label_map_root={args.closeup_label_map_root!r}"
            )

        # prints out image/label paths as a sanity check
        print("Close-up image paths:")
        for path in instance_data_root:
            print("-", path)
        
        print("Close-up label map paths:")
        for path in closeup_label_map_root:
            print("-", path)

        # --- INDIVIDUAL TEXT PROMPTS ---
        self.prompt_map = {}
        prompts_file_path = os.path.join(os.path.dirname(instance_data_root[0]), "prompts.txt")
        if os.path.exists(prompts_file_path):
            print(f"Loading individual prompts...")
            with open(prompts_file_path, "r") as f:
                for line in f:
                    filename, prompt = line.split(":", 1)
                    self.prompt_map[filename.strip()] = prompt.strip()
            self.custom_instance_prompts = True
        else:
            print("[WARNING] prompts.txt not found. Defaulting to universal instance_prompt for all images")
        
        self.zoom_ims_with_paths = [(Image.open(im_path), im_path) for im_path in instance_data_root]
        
        # load full training/validation images
        zoom_ims = [item[0] for item in self.zoom_ims_with_paths]
        full_im = load_image_correct_orientation(val_data_root)

        # load full training/validation label maps
        closeup_label_maps = [Image.open(path).convert("L") for path in closeup_label_map_root]
        full_label_map = Image.open(full_label_map_root).convert("L")

        # get's close-up image dimensions
        zoom_w, zoom_h = zoom_ims[0].size
        self.zoom_w, self.zoom_h = zoom_w, zoom_h

        # load full image mask
        mask_path = glob.glob(val_data_root.replace('.jpg', '_mask*.png'))
        assert len(mask_path) == 1, f'{os.path.basename(instance_data_root)} has {len(mask_path)} masks'
        mask_path = mask_path[0]

        print(f"Mask path: {mask_path}")

        full_mask = Image.open(mask_path).resize(full_im.size)
        full_mask = np.array(full_mask)
        full_mask = (full_mask/255).astype('uint8')  # convert to 0-1

        # debug sanity check to make sure masks line up
        print(f"Full image size: {full_im.size}")        # (width, height)
        print(f"Full mask shape: {full_mask.shape}")     # (height, width)

        # load scale for LR upsampling
        label_dir = os.path.dirname(instance_data_root[0]) + '/registration'
        scale_path = os.path.join(label_dir, 'scale.txt')
        scale = float(open(scale_path, 'r').readlines()[0].strip().split(': ')[1])

        # gap between video resolution (3840x2160) and closeup/full resolution (4032x)
        full_w, full_h = full_im.size
        if full_w > full_h:
            vid_w, vid_h = 3840, 2160
        else:
            vid_w, vid_h = 2160, 3840
        vid_w, vid_h = full_w, full_h

        gap_w, gap_h = full_w/vid_w, full_h/vid_h
        assert abs(gap_w - gap_h) < 1e-3
        scale /= gap_w
        scale = round(scale, 3)
        print(f'[INFO] Scale: {scale}')
        with open(os.path.join(test_vis_dir, 'final_scale.txt'), 'w') as f:
            f.write(f'{scale}\n')
        self.lr_size = lr_size = int(round(self.size / scale))

        # on the fly transformations (images)
        im_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        # on the fly transformations (label maps)
        label_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).long().unsqueeze(0))
            ]
        )

        from skimage.exposure import match_histograms

        # HUY
        if real_val_path:
            print(f"Loading real validation pairs from: {real_val_path}")
            lr_paths = sorted(glob.glob(os.path.join(real_val_path, "*_lr.jpg")))
            hr_paths = sorted(glob.glob(os.path.join(real_val_path, "*_hr.jpg")))

            if not lr_paths or not hr_paths:
                print("[WARNING] Could not find low-resolution/high-resolution validation pairs.")
            else:
                assert len(lr_paths) == len(hr_paths), "Mismatch in number of LR and HR validation images."

                val_real_lr_images = [Image.open(img).convert("RGB") for img in lr_paths]
                val_real_hr_images = [Image.open(img).convert("RGB") for img in hr_paths]

                rng = random.Random(42)

                closeup_x_min_list = []
                closeup_x_max_list = []
                closeup_y_min_list = []
                closeup_y_max_list = []

                warp_mask_path = os.path.join(label_dir, f'final_mask.png')
                warp_mask = Image.open(warp_mask_path).convert("L")

                # ensures correct orientation of mask and image
                warp_w, warp_h = warp_mask.size
                if (warp_h > warp_w) != (full_h > full_w):
                    warp_mask = warp_mask.rotate(270, expand=True)  # already rotated 90, rotate again to original orientation
                warp_mask = warp_mask.resize((full_w, full_h))

                # get bounding box of warped region 
                ys, xs = np.where(np.array(warp_mask) > 0)
                closeup_x_min, closeup_x_max = xs.min(), xs.max()+1
                closeup_y_min, closeup_y_max = ys.min(), ys.max()+1
                closeup_x_min_list.append(closeup_x_min)
                closeup_x_max_list.append(closeup_x_max)
                closeup_y_min_list.append(closeup_y_min)
                closeup_y_max_list.append(closeup_y_max)

                for i in range(len(lr_paths)):
                    
                    # loads first validation image
                    lr_img, hr_img = val_real_lr_images[i], val_real_hr_images[i]

                    ref = full_im.crop((closeup_x_min_list[0], closeup_y_min_list[0], closeup_x_max_list[0], closeup_y_max_list[0]))
                    ref_resized = mediapy.resize_image(np.array(ref), hr_img.size[::-1])
                    hr_img_before = np.array(hr_img)
                    hr_img = match_histograms(np.array(hr_img), ref_resized, channel_axis=-1)
                    mediapy.write_image(os.path.join(test_vis_dir, f'val_real_colormatch_{i}.png'), np.concatenate([hr_img_before, ref_resized, hr_img], axis=0))
                    hr_img = Image.fromarray(hr_img)

                    # Take a random, aligned crop from the LR and HR images
                    hr_w, hr_h = hr_img.size
                    lr_w, lr_h = lr_img.size

                    x_frac = rng.uniform(0, 1)
                    y_frac = rng.uniform(0, 1)

                    hr_i = int((hr_h - self.size) * y_frac)
                    hr_j = int((hr_w - self.size) * x_frac)
                    lr_i = int((lr_h - self.lr_size) * y_frac)
                    lr_j = int((lr_w - self.lr_size) * x_frac)

                    hr_i = min(hr_i, hr_h - self.size)
                    hr_j = min(hr_j, hr_w - self.size)
                    lr_i = min(lr_i, lr_h - self.lr_size)
                    lr_j = min(lr_j, lr_w - self.lr_size)

                    hr_crop = crop(hr_img, hr_i, hr_j, self.size, self.size)
                    lr_crop = crop(lr_img, lr_i, lr_j, self.lr_size, self.lr_size)

                    lr_crop_up = lr_crop.resize((self.size, self.size), resample=Image.BICUBIC)

                    self.val_real_hr_crops.append(im_transforms(hr_crop))
                    self.val_real_lr_crops.append(im_transforms(lr_crop_up))

                    hr_crop.save(os.path.join(test_vis_dir, f'val_real_hr_crop_{i}.png'))
                    lr_crop_up.save(os.path.join(test_vis_dir, f'val_real_lr_crop_up_{i}.png'))
                
         # for each closeup, color match to corresponding region in full image

        # training crops
        self.zoom_im_torch_list = []
        self.zoom_im_lr_torch_list = []
        self.train_images = []
        self.train_images_lr = []
        self.train_normals_lr = []

        # training label maps
        self.label_map_torch_list = []
        self.train_label_maps = []

        closeup_x_min_list = []
        closeup_x_max_list = []
        closeup_y_min_list = []
        closeup_y_max_list = []

        # validation crops
        self.val_images = []
        self.val_images_lr = []
        self.val_normals_lr = []
        
        # validation label maps
        self.val_label_maps = []
        self.val_label_maps_lr = []

        self.train_prompts = []

        # process each zoom image
        for closeup_idx, zoom_im in enumerate(zoom_ims):

            # load each corresponding warp mask
            warp_mask_path = os.path.join(label_dir, f'closeup{closeup_idx:02d}_final_mask.png')

            if not os.path.exists(warp_mask_path):
                warp_mask_path = os.path.join(label_dir, f'final_mask.png')

            warp_mask = Image.open(warp_mask_path).convert("L")

            # ensures correct orientation of mask and image
            warp_w, warp_h = warp_mask.size
            if (warp_h > warp_w) != (full_h > full_w):
                warp_mask = warp_mask.rotate(270, expand=True)  # already rotated 90, rotate again to original orientation
            warp_mask = warp_mask.resize((full_w, full_h))

            # get bounding box of warped region 
            ys, xs = np.where(np.array(warp_mask) > 0)
            closeup_x_min, closeup_x_max = xs.min(), xs.max()+1
            closeup_y_min, closeup_y_max = ys.min(), ys.max()+1
            closeup_x_min_list.append(closeup_x_min)
            closeup_x_max_list.append(closeup_x_max)
            closeup_y_min_list.append(closeup_y_min)
            closeup_y_max_list.append(closeup_y_max)

            # color matching
            ref = full_im.crop((closeup_x_min, closeup_y_min, closeup_x_max, closeup_y_max))
            ref_resized = mediapy.resize_image(np.array(ref), (zoom_h, zoom_w))
            if not args.ablation_name == 'ablation2':

                def reinhard_color_transfer(src, ref):
                    # Convert to LAB
                    src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.float32)
                    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2LAB).astype(np.float32)

                    # Compute channel means and stds
                    src_mean, src_std = cv2.meanStdDev(src_lab)
                    ref_mean, ref_std = cv2.meanStdDev(ref_lab)

                    # Apply normalization channel-wise
                    result = (src_lab - src_mean.T) / (src_std.T + 1e-6)
                    result = result * ref_std.T + ref_mean.T

                    # Clip + convert back
                    result = np.clip(result, 0, 255).astype(np.uint8)
                    result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
                    
                    return result
                
                zoom_im_before = np.array(zoom_im)
                zoom_im = reinhard_color_transfer(np.array(zoom_im), ref_resized)
                full_im_vis = np.array(full_im)
                full_im_vis = cv2.rectangle(full_im_vis, (closeup_x_min, closeup_y_min), (closeup_x_max, closeup_y_max), (0, 255, 0), 10)
                mediapy.write_image(os.path.join(test_vis_dir, f'closeup{closeup_idx:02d}_warp_region_vis.png'), full_im_vis)
                mediapy.write_image(os.path.join(test_vis_dir, f'closeup{closeup_idx:02d}_colormatch.png'), np.concatenate([zoom_im_before, ref_resized, zoom_im], axis=0)) 
                zoom_im = Image.fromarray(zoom_im)
                
            else:
                print(f'[INFO] Ablation: {args.ablation_name}')
                print(f'[INFO] No colormatch')

            # create lr image or load if normal control
            normal = None
            normal_val = None
            if not args.normal_control:
                
                # downsample ground truth and label map to create synthetic LR
                zoom_h_lr, zoom_w_lr = int(round(zoom_h/scale/args.extra_scale)), int(round(zoom_w/scale/args.extra_scale))
                zoom_im_lr = zoom_im.resize((zoom_w_lr, zoom_h_lr), resample=Image.BICUBIC)
                label_map_lr = closeup_label_maps[closeup_idx].resize((zoom_w_lr, zoom_h_lr), resample=Image.NEAREST)

                # optionally degrade LR image after downsampling
                if args.add_blur:
                    from PIL import ImageFilter
                    zoom_im_lr = zoom_im_lr.filter(ImageFilter.GaussianBlur(radius=args.blur_sigma))
                if args.jpeg_degradation:
                    import io
                    buffer = io.BytesIO()
                    zoom_im_lr.save(buffer, format="JPEG", quality=args.jpeg_quality)
                    buffer.seek(0)
                    zoom_im_lr = Image.open(buffer)
                
                # upsample LR back to HR resolution
                zoom_im_lr_up = zoom_im_lr.resize((zoom_w, zoom_h), resample=Image.BICUBIC)
                label_map_lr_up = label_map_lr.resize((zoom_w, zoom_h), resample=Image.NEAREST)

            else:
                assert False, 'normal control not supported'
                lr_path = instance_data_root.replace('_hr.jpg', '_lr.jpg')
                zoom_normal_lr_path = lr_path.replace('.jpg', '_mogenormals.png')
                full_normal_hr_path = val_data_root.replace('.jpg', '_mogenormals.png')
                zoom_im_lr = Image.open(lr_path)
                zoom_normal_lr = Image.open(zoom_normal_lr_path)
                full_normal_hr = Image.open(full_normal_hr_path)
                self.zoom_normal_lr_torch = im_transforms(zoom_normal_lr)
            
            # save normalized training closeup
            self.zoom_im_torch_list.append(im_transforms(zoom_im))
            self.zoom_im_lr_torch_list.append(im_transforms(zoom_im_lr_up))

            # save training label map as a tensor
            self.label_map_torch_list.append(label_transforms(label_map_lr_up))
            
            # number of training crops
            n_train = 1
            torch2pil = transforms.ToPILImage()

            for _ in range(n_train):

                # random crop location
                h_start = random.randint(0, self.zoom_h - self.size)
                w_start = random.randint(0, self.zoom_w - self.size)

                # add cropped image to training images
                self.train_images.append(self.zoom_im_torch_list[closeup_idx][:, h_start:h_start+self.size, w_start:w_start+self.size])
                self.train_images_lr.append(self.zoom_im_lr_torch_list[closeup_idx][:, h_start:h_start+self.size, w_start:w_start+self.size])
                
                # add cropped label map to training label maps
                self.train_label_maps.append(self.label_map_torch_list[closeup_idx][:, h_start:h_start+self.size, w_start:w_start+self.size])
                
                # --- INDIVIDUAL TEXT PROMPTS ---
                img_path = self.zoom_ims_with_paths[closeup_idx][1]
                img_filename = os.path.basename(img_path)
                prompt = self.prompt_map.get(img_filename, self.instance_prompt)
                self.train_prompts.append(prompt)
                
                if self.normal_control:
                    self.train_normals_lr.append(self.zoom_normal_lr_torch_list[closeup_idx][:, h_start:h_start+self.size, w_start:w_start+self.size])

                # Save ground truth crops for debugging
                torch2pil((self.train_images[-1]+1)/2).save(os.path.join(test_vis_dir, f'train_lr_gt_{len(self.train_images)}.png'))
                torch2pil((self.train_images_lr[-1]+1)/2).save(os.path.join(test_vis_dir, f'train_lr_up_{len(self.train_images)}.png'))

                # Save ground truth label map for debugging
                torch2pil((self.train_label_maps[-1]+1)/2).save(os.path.join(test_vis_dir, f'train_label_map_lr_{len(self.train_label_maps)}.png'))

                # Save raw LR crop
                lr_h_start, lr_w_start = int(round(h_start/scale/args.extra_scale)), int(round(w_start/scale/args.extra_scale))
                zoom_im_lr.crop((lr_w_start, lr_h_start, lr_w_start+lr_size//args.extra_scale, lr_h_start+lr_size//args.extra_scale)).save(os.path.join(test_vis_dir, f'train_lr_{len(self.train_images)}.png'))
                if self.normal_control:
                    torch2pil((self.train_normals_lr[-1]+1)/2).save(os.path.join(test_vis_dir, f'train_lr_up_normal_{len(self.train_images)}.png'))

                # Visualize crop location
                zoom_im_vis = np.array(zoom_im)
                zoom_im_vis = cv2.rectangle(zoom_im_vis, (w_start, h_start), (w_start+self.size, h_start+self.size), (0, 0, 255), 10)
                zoom_im_vis = Image.fromarray(zoom_im_vis)
                zoom_im_vis.save(os.path.join(test_vis_dir, f'train_lr_viscrop_{len(self.train_images)}.png'))

            # validation crop prep
            x_start = (closeup_x_min+closeup_x_max)//2-lr_size//2
            y_start = (closeup_y_min+closeup_y_max)//2-lr_size//2
            full_ys, full_xs = np.where(full_mask)
            full_x_min, full_y_min = full_xs.min(), full_ys.min()
            full_x_max, full_y_max = full_xs.max(), full_ys.max()

            # wandb, validation crops
            # 3 samples
            n_val = 2 * (closeup_idx+1)
            max_run = n_val * 5
            run_idx = 0
            while len(self.val_images) < n_val and run_idx < max_run:

                # get the first crop from the center, the rest random
                if run_idx == 0:
                    x_start_val = x_start
                    y_start_val = y_start
                else:
                    x_start_val = random.randint(full_x_min, full_x_max-lr_size)
                    y_start_val = random.randint(full_y_min, full_y_max-lr_size)
                
                # crop coordinates
                x_end_val = x_start_val + lr_size
                y_end_val = y_start_val + lr_size

                # mask region of current crop, make sure validation image contains at least 90% train material
                patch_mask = full_mask[y_start_val:y_end_val, x_start_val:x_end_val].astype(float)
                if np.mean(patch_mask) > 0.9:

                    # get low-res validation crop and upscale
                    lr = full_im.crop((x_start_val, y_start_val, x_end_val, y_end_val))
                    lr_up = lr.resize((size, size), resample=Image.BICUBIC)

                    # get low-res validation label map and upscale
                    label_map_lr = full_label_map.crop((x_start_val, y_start_val, x_end_val, y_end_val))
                    label_map_lr_up = label_map_lr.resize((size, size), resample=Image.NEAREST)

                    if self.normal_control:
                        lr_up_normal = full_normal_hr.crop((x_start_val, y_start_val, x_end_val, y_end_val)).resize((size, size), resample=Image.NEAREST)

                    # save normalized upscaled validation crop
                    self.val_images_lr.append(im_transforms(lr_up))
                    self.val_images.append(None)

                    # save normalized upscaled validation label map
                    self.val_label_maps_lr.append(label_transforms(label_map_lr_up))
                    self.val_label_maps.append(None)

                    lr_up.save(os.path.join(test_vis_dir, f'test_lr_up_{len(self.val_images)}.png'))
                    lr.save(os.path.join(test_vis_dir, f'test_lr_{len(self.val_images)}.png'))
                    label_map_lr_up.save(os.path.join(test_vis_dir, f'test_label_map_lr_up_{len(self.val_images)}.png'))
                    if self.normal_control:
                        self.val_normals_lr.append(im_transforms(lr_up_normal))
                        lr_up_normal.save(os.path.join(test_vis_dir, f'test_lr_up_normal_{len(self.val_images)}.png'))
                
                    # box crop location
                    full_im_vis = np.array(full_im)
                    full_im_vis = cv2.rectangle(full_im_vis, (x_start_val, y_start_val), (x_end_val, y_end_val), (0, 0, 255), 10)
                    full_im_vis = Image.fromarray(full_im_vis)
                    full_im_vis.save(os.path.join(test_vis_dir, f'test_lr_viscrop_{len(self.val_images)}.png'))

                run_idx += 1

        if args.skip_training:
            import sys; sys.exit()

    # helper method to return length... lol.
    def __len__(self):
        return self._length

    # helper method to return crops/prompts for training
    def __getitem__(self, index):

        example = {}

        # get random crop dimensions
        h_start = random.randint(0, self.zoom_h - self.size)
        w_start = random.randint(0, self.zoom_w - self.size)
        
        # get a random training close-up
        img_idx = random.randint(0, len(self.zoom_im_torch_list)-1)

        img_path = self.zoom_ims_with_paths[img_idx][1]
        img_filename = os.path.basename(img_path)

        prompt = self.prompt_map.get(img_filename, self.instance_prompt)

        augmentations = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ])

        hr_image_crop = self.zoom_im_torch_list[img_idx][:, h_start:h_start+self.size, w_start:w_start+self.size]
        lr_image_crop = self.zoom_im_lr_torch_list[img_idx][:, h_start:h_start+self.size, w_start:w_start+self.size]
        label_map_crop = self.label_map_torch_list[img_idx][:, h_start:h_start+self.size, w_start:w_start+self.size]

        stacked_channels = torch.cat([hr_image_crop, lr_image_crop, label_map_crop], dim=0)
        transformed_stack = augmentations(stacked_channels)

        transformed_hr, transformed_lr, transformed_label_map = torch.chunk(transformed_stack, chunks=3, dim=0)

        # find all prompts/crops
        example["instance_prompt"] = prompt
        example["instance_images"] = transformed_hr
        example["instance_images_lr"] = transformed_lr
        example["instance_label_map"] = transformed_label_map

        if self.normal_control:
            example["instance_normals_lr"] = self.zoom_normal_lr_torch_list[img_idx][:, h_start:h_start+self.size, w_start:w_start+self.size]

        return example

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


def tokenize_prompt(tokenizer, prompt, max_sequence_length):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    return text_input_ids


def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length=512,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    text_input_ids=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    text_input_ids=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

    # Use pooled output of CLIPTextModel
    prompt_embeds = prompt_embeds.pooler_output
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

    return prompt_embeds


def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
    text_input_ids_list=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    dtype = text_encoders[0].dtype

    pooled_prompt_embeds = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        device=device if device is not None else text_encoders[0].device,
        num_images_per_prompt=num_images_per_prompt,
        text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
    )

    prompt_embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[1].device,
        text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
    )

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

    return prompt_embeds, pooled_prompt_embeds, text_ids


def main(args):

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    train_dataset = OnlineUpscalerDreamBoothDataset(
        args=args,
        instance_data_root=args.instance_data_dir,
        val_data_root=args.val_data_dir,
        test_vis_dir=args.test_vis_dir,
        instance_prompt=args.instance_prompt,
        size=args.resolution,
        closeup_label_map_root=args.closeup_label_map_root,
        full_label_map_root=args.full_label_map_root,
        real_val_path=args.real_val_path
    )

    print(f'[INFO] Dataset loaded')

    if args.train_text_encoder:
        assert False, "This script does not support training text encoder yet."

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Generate class images if prior preservation is enabled.
    if args.with_prior_preservation:
        class_images_dir = Path(args.class_data_dir)
        if not class_images_dir.exists():
            class_images_dir.mkdir(parents=True)
        cur_class_images = len(list(class_images_dir.iterdir()))

        if cur_class_images < args.num_class_images:
            has_supported_fp16_accelerator = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.float16 if has_supported_fp16_accelerator else torch.float32
            if args.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif args.prior_generation_precision == "fp16":
                torch_dtype = torch.float16
            elif args.prior_generation_precision == "bf16":
                torch_dtype = torch.bfloat16
            pipeline = CustomFluxControlNetPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=args.revision,
                variant=args.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            num_new_images = args.num_class_images - cur_class_images
            logger.info(f"Number of class images to sample: {num_new_images}.")

            sample_dataset = PromptDataset(args.class_prompt, num_new_images)
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

            sample_dataloader = accelerator.prepare(sample_dataloader)
            pipeline.to(accelerator.device)

            for example in tqdm(
                sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
            ):
                images = pipeline(example["prompt"]).images

                for i, image in enumerate(images):
                    hash_image = insecure_hashlib.sha1(image.tobytes()).hexdigest()
                    image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                    image.save(image_filename)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
            ).repo_id

    # Load the tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )  # 83M parameters
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )  # 11.9B parameters
    controlnet = FluxControlNetModel.from_pretrained(
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
    )  # 1.8B parameters

    # We only train the additional adapter LoRA layers
    transformer.requires_grad_(False)
    controlnet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
    
    # Expand the transformer's patch-embedding linear to accept label map channels appended to
    # the noisy latent. New weights for the extra channels are zero-initialized so training begins
    # from the pretrained behavior; the label map signal is learned on top incrementally.
    with torch.no_grad():
        initial_in_features = transformer.x_embedder.in_features
        initial_input_channels = transformer.config.in_channels
        print(f"Original input dimensions: {initial_input_channels}")
        new_linear = torch.nn.Linear(
            initial_in_features + (args.num_textures * 4),
            transformer.x_embedder.out_features,
            bias=transformer.x_embedder.bias is not None,
            dtype=transformer.dtype,
            device=transformer.device,
        )
        new_linear.weight.zero_()
        new_linear.weight[:, :initial_in_features].copy_(transformer.x_embedder.weight)

        if transformer.x_embedder.bias is not None:
            new_linear.bias.copy_(transformer.x_embedder.bias)
        transformer.x_embedder = new_linear

    assert torch.all(transformer.x_embedder.weight[:, initial_input_channels:].data == 0)
    transformer.register_to_config(in_channels=initial_input_channels + (args.num_textures * 4), out_channels=initial_input_channels)
    print(f"New input dimensions: {transformer.config.in_channels}")

    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            "x_embedder",
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]

    # Add LoRA adapters to both the transformer and the ControlNet. Sharing the same config
    # keeps their rank and alpha consistent; the ControlNet adapter learns upscaler-specific
    # residuals while the transformer adapter steers texture synthesis via the label map.
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)
    controlnet.add_adapter(transformer_lora_config)

    if args.train_text_encoder:
        text_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_lora_checkpoint(save_directory, transformer_lora_layers, controlnet_lora_layers, text_encoder_lora_layers):
        signature = inspect.signature(CustomFluxControlNetPipeline.save_lora_weights)
        save_kwargs = {}

        if "transformer_lora_layers" in signature.parameters and transformer_lora_layers is not None:
            save_kwargs["transformer_lora_layers"] = transformer_lora_layers
        if "text_encoder_lora_layers" in signature.parameters and text_encoder_lora_layers is not None:
            save_kwargs["text_encoder_lora_layers"] = text_encoder_lora_layers
        if "controlnet_lora_layers" in signature.parameters and controlnet_lora_layers is not None:
            save_kwargs["controlnet_lora_layers"] = controlnet_lora_layers

        CustomFluxControlNetPipeline.save_lora_weights(save_directory=save_directory, **save_kwargs)

        controlnet_path = os.path.join(save_directory, CONTROLNET_LORA_WEIGHTS_NAME)
        if controlnet_lora_layers is not None and "controlnet_lora_layers" not in signature.parameters:
            logger.info(
                "Installed diffusers save_lora_weights does not support controlnet_lora_layers; "
                "saving ControlNet LoRA separately to %s",
                controlnet_path,
            )
            torch.save(controlnet_lora_layers, controlnet_path)
        elif os.path.exists(controlnet_path):
            os.remove(controlnet_path)

    def load_controlnet_adapter(model, input_dir):
        controlnet_path = os.path.join(input_dir, CONTROLNET_LORA_WEIGHTS_NAME)
        if not os.path.isfile(controlnet_path):
            return False

        controlnet_state_dict = torch.load(controlnet_path, map_location="cpu")
        if not controlnet_state_dict:
            return False

        if not hasattr(model, "peft_config") or "default" not in model.peft_config:
            model.add_adapter(transformer_lora_config)

        controlnet_state_dict = convert_unet_state_dict_to_peft(controlnet_state_dict)
        incompatible_keys = set_peft_model_state_dict(model, controlnet_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    "Loading ControlNet LoRA weights from %s produced unexpected keys: %s",
                    controlnet_path,
                    unexpected_keys,
                )
        return True

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            controlnet_lora_layers_to_save = None
            text_encoder_one_lora_layers_to_save = None

            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(unwrap_model(controlnet))):
                    controlnet_lora_layers_to_save = get_peft_model_state_dict(model)
                elif isinstance(model, type(unwrap_model(text_encoder_one))):
                    text_encoder_one_lora_layers_to_save = get_peft_model_state_dict(model)
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

            save_lora_checkpoint(
                save_directory=output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save,
                controlnet_lora_layers=controlnet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_one_lora_layers_to_save,
            )

    def load_model_hook(models, input_dir):
        transformer_ = None
        controlnet_ = None
        text_encoder_one_ = None

        while len(models) > 0:
            model = models.pop()

            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            elif isinstance(model, type(unwrap_model(controlnet))):
                controlnet_ = model
            elif isinstance(model, type(unwrap_model(text_encoder_one))):
                text_encoder_one_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = CustomFluxControlNetPipeline.lora_state_dict(input_dir)

        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )

        # repeat for controlnet
        controlnet_loaded = load_controlnet_adapter(controlnet_, input_dir)
        if not controlnet_loaded:
            controlnet_state_dict = {
                f'{k.replace("controlnet.", "")}': v for k, v in lora_state_dict.items() if k.startswith("controlnet.")
            }
            controlnet_state_dict = convert_unet_state_dict_to_peft(controlnet_state_dict)
            incompatible_keys = set_peft_model_state_dict(controlnet_, controlnet_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        if args.train_text_encoder:
            # Do we need to call `scale_lora_layers()` here?
            _set_state_dict_into_text_encoder(lora_state_dict, prefix="text_encoder.", text_encoder=text_encoder_one_)

        # Make sure the trainable params are in float32. This is again needed since the base models
        # are in `weight_dtype`. More details:
        # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
        if args.mixed_precision == "fp16":
            models = [transformer_, controlnet_]
            if args.train_text_encoder:
                models.extend([text_encoder_one_])
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(models)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [transformer, controlnet]
        if args.train_text_encoder:
            models.extend([text_encoder_one])
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))  # 11M parameters
    controlnet_lora_parameters = list(filter(lambda p: p.requires_grad, controlnet.parameters()))  # 2.2M parameters
    if args.train_text_encoder:
        text_lora_parameters_one = list(filter(lambda p: p.requires_grad, text_encoder_one.parameters()))

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
    controlnet_parameters_with_lr = {"params": controlnet_lora_parameters, "lr": args.learning_rate}
    if args.train_text_encoder:
        # different learning rate for text encoder and unet
        text_parameters_one_with_lr = {
            "params": text_lora_parameters_one,
            "weight_decay": args.adam_weight_decay_text_encoder,
            "lr": args.text_encoder_lr if args.text_encoder_lr else args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr, text_parameters_one_with_lr]
    else:
        params_to_optimize = [transformer_parameters_with_lr, controlnet_parameters_with_lr]

    # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if args.use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )
        if args.train_text_encoder and args.text_encoder_lr:
            logger.warning(
                f"Learning rates were provided both for the transformer and the text encoder- e.g. text_encoder_lr:"
                f" {args.text_encoder_lr} and learning_rate: {args.learning_rate}. "
                f"When using prodigy only learning_rate is used as the initial learning rate."
            )
            # changes the learning rate of text_encoder_parameters_one to be
            # --learning_rate
            params_to_optimize[1]["lr"] = args.learning_rate

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    if not args.train_text_encoder:
        tokenizers = [tokenizer_one, tokenizer_two]
        text_encoders = [text_encoder_one, text_encoder_two]

        def compute_text_embeddings(prompt, text_encoders, tokenizers):
            with torch.no_grad():
                prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                    text_encoders, tokenizers, prompt, args.max_sequence_length
                )
                prompt_embeds = prompt_embeds.to(accelerator.device)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
                text_ids = text_ids.to(accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # If no type of tuning is done on the text_encoder and custom instance prompts are NOT
    # provided (i.e. the --instance_prompt is used for all images), we encode the instance prompt once to avoid
    # the redundant encoding.
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    # Handle class prompt for prior-preservation.
    if args.with_prior_preservation:
        if not args.train_text_encoder:
            class_prompt_hidden_states, class_pooled_prompt_embeds, class_text_ids = compute_text_embeddings(
                args.class_prompt, text_encoders, tokenizers
            )

    # Clear the memory here
    if not args.train_text_encoder and not train_dataset.custom_instance_prompts:
        del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
        free_memory()

    # If custom instance prompts are NOT provided (i.e. the instance prompt is used for all images),
    # pack the statically computed variables appropriately here. This is so that we don't
    # have to pass them to the dataloader.

    if not train_dataset.custom_instance_prompts:
        if not args.train_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids
            if args.with_prior_preservation:
                prompt_embeds = torch.cat([prompt_embeds, class_prompt_hidden_states], dim=0)
                pooled_prompt_embeds = torch.cat([pooled_prompt_embeds, class_pooled_prompt_embeds], dim=0)
                text_ids = torch.cat([text_ids, class_text_ids], dim=0)
        # if we're optimizing the text encoder (both if instance prompt is used for all images or custom prompts)
        # we need to tokenize and encode the batch prompts on all training steps
        else:
            tokens_one = tokenize_prompt(tokenizer_one, args.instance_prompt, max_sequence_length=77)
            tokens_two = tokenize_prompt(
                tokenizer_two, args.instance_prompt, max_sequence_length=args.max_sequence_length
            )
            if args.with_prior_preservation:
                class_tokens_one = tokenize_prompt(tokenizer_one, args.class_prompt, max_sequence_length=77)
                class_tokens_two = tokenize_prompt(
                    tokenizer_two, args.class_prompt, max_sequence_length=args.max_sequence_length
                )
                tokens_one = torch.cat([tokens_one, class_tokens_one], dim=0)
                tokens_two = torch.cat([tokens_two, class_tokens_two], dim=0)

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    accelerator.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

        if args.validation_prompt is None:
            del vae
            free_memory()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    if args.train_text_encoder:
        (
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            transformer,
            text_encoder_one,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )
    else:
        transformer, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            transformer, controlnet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        init_kwargs = {}
        if args.report_to in ("wandb", "all") and args.wandb_run_name:
            init_kwargs["wandb"] = {"name": args.wandb_run_name}
        accelerator.init_trackers(args.tracker_name, config=vars(args), init_kwargs=init_kwargs)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1].replace('ep', '')))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_epoch = int(path.split("-")[1].replace('ep', ''))+1
            global_step = global_epoch * num_update_steps_per_epoch

            initial_global_step = global_step
            first_epoch = global_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    with open(os.path.join(args.output_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")

    # training loop
    for epoch in range(first_epoch, args.num_train_epochs):

        # Two-stage schedule: the first half of training uses only label-map guidance so the
        # transformer learns multi-texture control before ControlNet upscaling is introduced.
        # In the second half (stage1=False) the ControlNet is active, adding super-resolution
        # guidance on top of the already-learned label-map conditioning.
        stage1 = (epoch < args.num_train_epochs // 2)

        # set models into training mode
        transformer.train()
        controlnet.train()

        if args.train_text_encoder:
            text_encoder_one.train()
            # set top parameter requires_grad = True for gradient checkpointing works
            accelerator.unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer, controlnet]
            if args.train_text_encoder:
                models_to_accumulate.extend([text_encoder_one])
            with accelerator.accumulate(models_to_accumulate):
                prompts = batch["prompts"]

                # encode batch prompts when custom prompts are provided for each image -
                if train_dataset.custom_instance_prompts:
                    if not args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
                            prompts, text_encoders, tokenizers
                        )
                    else:
                        tokens_one = tokenize_prompt(tokenizer_one, prompts, max_sequence_length=77)
                        tokens_two = tokenize_prompt(
                            tokenizer_two, prompts, max_sequence_length=args.max_sequence_length
                        )
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[tokens_one, tokens_two],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=prompts,
                        )
                else:
                    elems_to_repeat = len(prompts)
                    if args.train_text_encoder:
                        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
                            text_encoders=[text_encoder_one, text_encoder_two],
                            tokenizers=[None, None],
                            text_input_ids_list=[
                                tokens_one.repeat(elems_to_repeat, 1),
                                tokens_two.repeat(elems_to_repeat, 1),
                            ],
                            max_sequence_length=args.max_sequence_length,
                            device=accelerator.device,
                            prompt=args.instance_prompt,
                        )

                # Convert images to latent space
                if args.cache_latents:
                    concatenated_noisy_model_input = latents_cache[step].sample()
                else:
                    # flux latents
                    pixel_latents = vae.encode(batch["pixel_values"].to(dtype=vae.dtype)).latent_dist.sample() # (B, 16, 128, 128)

                    # Convert the integer label map to a one-hot tensor at latent resolution.
                    # Each channel corresponds to one texture class; nearest-neighbor downsampling
                    # preserves hard boundaries between texture regions.
                    grayscale_map = batch["pixel_values_label_map"].to(torch.long) # (B, 1, H, W)

                    latent_height = pixel_latents.shape[2]
                    latent_width = pixel_latents.shape[3]

                    label_map_downsampled = torch.nn.functional.interpolate(
                        grayscale_map.float(),
                        size = (latent_height, latent_width),
                        mode='nearest'
                    ).long().squeeze(1)

                    label_map_one_hot = torch.nn.functional.one_hot(
                        label_map_downsampled,
                        num_classes=args.num_textures,
                    ).permute(0, 3, 1, 2).to(torch.float16) # (B, num_textures, H/8, W/8)
                    
                    # controlnet latents
                    pixel_values_lr = batch["pixel_values_lr"].to(dtype=vae.dtype)
                    model_input_lr = vae.encode(pixel_values_lr).latent_dist.sample()

                # flux latents shift
                pixel_latents = (pixel_latents - vae_config_shift_factor) * vae_config_scaling_factor
                pixel_latents = pixel_latents.to(dtype=weight_dtype)

                # controlnet latents shift
                model_input_lr = (model_input_lr - vae_config_shift_factor) * vae_config_scaling_factor
                model_input_lr = model_input_lr.to(dtype=weight_dtype)

                vae_scale_factor = 2 ** (len(vae_config_block_out_channels) - 1)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(pixel_latents)
                bsz = pixel_latents.shape[0]

                # Sample timesteps with optional non-uniform weighting.
                # 'logit_normal' biases toward mid-range timesteps (better perceptual detail);
                # 'mode' concentrates around a single peak; 'cosmap' applies cosine reweighting.
                # Uniform sampling is used when weighting_scheme='uniform'.
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # Flow matching interpolation: zt = (1 - sigma) * x0 + sigma * noise.
                # The one-hot label map is appended as extra channels so the transformer sees
                # which texture region each latent patch belongs to at every denoising step.
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                concatenated_noisy_model_input = torch.cat([noisy_model_input, label_map_one_hot], dim=1)

                packed_control_image = CustomFluxControlNetPipeline._pack_latents(
                    model_input_lr,
                    batch_size=model_input_lr.shape[0],
                    num_channels_latents=model_input_lr.shape[1],
                    height=model_input_lr.shape[2],
                    width=model_input_lr.shape[3],
                )

                packed_noisy_model_input_transformer = CustomFluxControlNetPipeline._pack_latents(
                    concatenated_noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=concatenated_noisy_model_input.shape[1],
                    height=concatenated_noisy_model_input.shape[2],
                    width=concatenated_noisy_model_input.shape[3],
                )

                packed_noisy_model_input_controlnet = CustomFluxControlNetPipeline._pack_latents(
                    noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=noisy_model_input.shape[1],
                    height=noisy_model_input.shape[2],
                    width=noisy_model_input.shape[3],
                )

                latent_image_ids = CustomFluxControlNetPipeline._prepare_latent_image_ids(
                    bsz,
                    concatenated_noisy_model_input.shape[2] // 2,
                    concatenated_noisy_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # handle guidance
                if accelerator.unwrap_model(transformer).config.guidance_embeds:  # True
                    guidance = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=noisy_model_input.device,
                        dtype=weight_dtype
                    )
                else:
                    guidance = None

                # Skip ControlNet in stage 1 so the transformer first learns label-map conditioning
                # in isolation. Passing None for the block samples disables the residual injection.
                if stage1:
                    controlnet_block_samples, controlnet_single_block_samples = None, None
                else:
                    controlnet_block_samples, controlnet_single_block_samples = controlnet(
                        hidden_states=packed_noisy_model_input_controlnet,
                        controlnet_cond=packed_control_image,
                        controlnet_mode=None,
                        conditioning_scale=args.controlnet_conditioning_scale,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        pooled_projections=pooled_prompt_embeds,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=None,
                        return_dict=False,
                    )

                # a100, 1024x1024, 53217MB before (after val), 40513 before val
                # import pdb; pdb.set_trace()

                # 1. For the first half of training, do not use the ControlNet inputs or update ControlNet blocks (we want to ensure the label map is being used first)
                model_pred = transformer(
                    hidden_states=packed_noisy_model_input_transformer,
                    # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                    joint_attention_kwargs=None,
                    controlnet_blocks_repeat=False,
                )[0]

                model_pred = CustomFluxControlNetPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # Flow matching target: the model predicts the vector field v = noise - x0.
                target = noise - pixel_latents

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute prior loss
                    prior_loss = torch.mean(
                        (weighting.float() * (model_pred_prior.float() - target_prior.float()) ** 2).reshape(
                            target_prior.shape[0], -1
                        ),
                        1,
                    )
                    prior_loss = prior_loss.mean()

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                if args.with_prior_preservation:
                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        itertools.chain(transformer.parameters(), text_encoder_one.parameters())
                        if args.train_text_encoder
                        else transformer.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        if accelerator.is_main_process:
            checkpoint_dir = args.output_dir
            if epoch % args.checkpointing_epochs == 0:
                # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1].replace('ep', '')))

                    # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        removing_checkpoints = checkpoints[0:num_to_remove]

                        logger.info(
                            f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        )
                        logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        for removing_checkpoint in removing_checkpoints:
                            removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                            shutil.rmtree(removing_checkpoint)

                save_path = os.path.join(args.output_dir, f"checkpoint-ep{epoch:04d}")
                logger.info(f"Starting checkpoint save to {save_path}")
                try:
                    accelerator.save_state(save_path)
                except Exception:
                    logger.exception("Checkpoint save failed for %s. Validation will be skipped.", save_path)
                    raise
                logger.info(f"Checkpoint save complete: {save_path}")
                checkpoint_dir = save_path

            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(f"Starting validation for epoch {epoch} using checkpoint directory {checkpoint_dir}")
                # create pipeline
                if not args.train_text_encoder:
                    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)
                    text_encoder_one.to(weight_dtype)
                    text_encoder_two.to(weight_dtype)
                
                pipeline = CustomFluxControlNetPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    vae=vae,
                    text_encoder=accelerator.unwrap_model(text_encoder_one),
                    text_encoder_2=accelerator.unwrap_model(text_encoder_two),
                    transformer=accelerator.unwrap_model(transformer),
                    controlnet=accelerator.unwrap_model(controlnet),
                    revision=args.revision,
                    variant=args.variant,
                    torch_dtype=weight_dtype,
                )
                pipeline_args = {"prompt": args.validation_prompt, "controlnet_conditioning_scale": args.controlnet_conditioning_scale, "num_inference_steps": 28, "guidance_scale": 3.5, "height": args.resolution, "width": args.resolution}

                # import pdb; pdb.set_trace()
                # a40
                # 192x192, 41511/46068MiB
                # 128x128, 41139/46068MiB
                # a100
                # 64x64, 41163/81920MiB, after loading pipeline to gpu, 50321/81920MiB
                # 128x128, 41645 before, 50803 after
                # 1024x1024, 74833 before, 78427 during generation
                # import pdb; pdb.set_trace()
                images = log_validation(
                    pipeline=pipeline,
                    args=args,
                    accelerator=accelerator,
                    pipeline_args=pipeline_args,
                    epoch=epoch,
                    torch_dtype=weight_dtype,
                    groundtruth=train_dataset.val_images,
                    control_image=train_dataset.val_images_lr,
                    label_maps_val=train_dataset.val_label_maps_lr,
                    groundtruth_train=train_dataset.train_images,
                    control_image_train=train_dataset.train_images_lr,
                    label_maps_train=train_dataset.train_label_maps,
                    lr_size=train_dataset.lr_size,
                    hr_size=train_dataset.size,

                    # huy
                    val_real_lr_crops=train_dataset.val_real_lr_crops,
                    val_real_hr_crops=train_dataset.val_real_hr_crops,
                    train_prompts=train_dataset.train_prompts,
                    stage1=stage1,
                )
                if not args.train_text_encoder:
                    del text_encoder_one, text_encoder_two
                    free_memory()

                images = None
                del pipeline
                free_memory()
                logger.info(f"Validation complete for epoch {epoch}")

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        
        unwrap_model(transformer)

        if args.upcast_before_saving:
            transformer.to(torch.float32)
        else:
            transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)

        if args.train_text_encoder:
            text_encoder_one = unwrap_model(text_encoder_one)
            text_encoder_lora_layers = get_peft_model_state_dict(text_encoder_one.to(torch.float32))
        else:
            text_encoder_lora_layers = None

        save_lora_checkpoint(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
            controlnet_lora_layers=get_peft_model_state_dict(unwrap_model(controlnet)),
            text_encoder_lora_layers=text_encoder_lora_layers,
        )
        free_memory()

        # Final inference
        # Load previous pipeline
        controlnet = FluxControlNetModel.from_pretrained(
        "jasperai/Flux.1-dev-Controlnet-Upscaler",
        torch_dtype=torch.bfloat16
        )
        pipeline = CustomFluxControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        controlnet=controlnet,
        torch_dtype=torch.bfloat16
        ).to("cuda")
        # load attention processors
        pipeline.load_lora_weights(args.output_dir)
        load_controlnet_adapter(pipeline.controlnet, args.output_dir)

        # run inference
        images = []
        if args.validation_prompt and args.num_validation_images > 0:
            pipeline_args = {"prompt": args.validation_prompt, "controlnet_conditioning_scale": args.controlnet_conditioning_scale, "num_inference_steps": 28, "guidance_scale": 3.5, "height": args.resolution, "width": args.resolution}
            images = log_validation(
                pipeline=pipeline,
                args=args,
                accelerator=accelerator,
                pipeline_args=pipeline_args,
                epoch=epoch,
                is_final_validation=True,
                torch_dtype=weight_dtype,
                groundtruth=train_dataset.val_images,
                control_image=train_dataset.val_images_lr,
                label_maps_val=train_dataset.val_label_maps_lr,
                groundtruth_train=train_dataset.train_images,
                control_image_train=train_dataset.train_images_lr,
                label_maps_train=train_dataset.train_label_maps,

                # huy
                val_real_lr_crops=train_dataset.val_real_lr_crops,
                val_real_hr_crops=train_dataset.val_real_hr_crops,
                stage1=stage1,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                train_text_encoder=args.train_text_encoder,
                instance_prompt=args.instance_prompt,
                validation_prompt=args.validation_prompt,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

        images = None
        del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
