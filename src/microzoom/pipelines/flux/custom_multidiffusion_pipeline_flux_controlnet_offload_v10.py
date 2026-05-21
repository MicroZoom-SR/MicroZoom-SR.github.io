# Copyright 2024 Black Forest Labs, The HuggingFace Team and The InstantX Team. All rights reserved.
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
# limitations under the License.

import time
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import os
import glob
import psutil
import numpy as np
import torch
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput
from torchvision.utils import save_image


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers.utils import load_image
        >>> from diffusers import FluxControlNetPipeline
        >>> from diffusers import FluxControlNetModel

        >>> controlnet_model = "InstantX/FLUX.1-dev-controlnet-canny"
        >>> controlnet = FluxControlNetModel.from_pretrained(controlnet_model, torch_dtype=torch.bfloat16)
        >>> pipe = FluxControlNetPipeline.from_pretrained(
        ...     base_model, controlnet=controlnet, torch_dtype=torch.bfloat16
        ... )
        >>> pipe.to("cuda")
        >>> control_image = load_image("https://huggingface.co/InstantX/SD3-Controlnet-Canny/resolve/main/canny.jpg")
        >>> prompt = "A girl in city, 25 years old, cool, futuristic"
        >>> image = pipe(
        ...     prompt,
        ...     control_image=control_image,
        ...     control_guidance_start=0.2,
        ...     control_guidance_end=0.8,
        ...     controlnet_conditioning_scale=1.0,
        ...     num_inference_steps=28,
        ...     guidance_scale=3.5,
        ... ).images[0]
        >>> image.save("flux.png")
        ```
"""


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.16,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxControlNetMultiDiffusionPipeline(DiffusionPipeline, FluxLoraLoaderMixin, FromSingleFileMixin):
    r"""
    The Flux pipeline for text-to-image generation.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
        controlnet: Union[
            FluxControlNetModel, List[FluxControlNetModel], Tuple[FluxControlNetModel], FluxMultiControlNetModel
        ],
        num_textures: int = 3,
    ):
        super().__init__()
        if isinstance(controlnet, (list, tuple)):
            controlnet = FluxMultiControlNetModel(controlnet)

        # vae = torch.compile(vae)
        # transformer = torch.compile(transformer)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
            controlnet=controlnet,
        )

        with torch.no_grad():
            transformer = self.transformer

            initial_in_features = transformer.x_embedder.in_features
            initial_input_channels = transformer.config.in_channels
            new_linear = torch.nn.Linear(
                initial_in_features + (num_textures * 4),
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
        
        transformer.register_to_config(in_channels=initial_input_channels + (num_textures * 4), out_channels=initial_input_channels)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        # Flux latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = (
            self.tokenizer.model_max_length if hasattr(self, "tokenizer") and self.tokenizer is not None else 77
        )
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]

        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer_max_length} tokens: {removed_text}"
            )
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, FluxLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)
            if self.text_encoder_2 is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder_2, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            # We only use the pooled prompt output from the CLIPTextModel
            pooled_prompt_embeds = self._get_clip_prompt_embeds(
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt_2,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        if self.text_encoder is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder, lora_scale)

        if self.text_encoder_2 is not None:
            if isinstance(self, FluxLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(self.text_encoder_2, lora_scale)

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def check_inputs(
        self,
        base_prompt_format,
        prompt_fragments,
        fall_back_prompt,
        height,
        width,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if base_prompt_format is None or prompt_fragments is None or fall_back_prompt is None:
            raise ValueError(
                "When using dynamic prompts, `base_prompt_format`, `prompt_fragments`, "
                "and `fall_back_prompt` must all be provided."
            )

        if prompt_embeds is not None and pooled_prompt_embeds is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `pooled_prompt_embeds` also have to be passed. Make sure to generate `pooled_prompt_embeds` from the same text encoder that was used to generate `prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._prepare_latent_image_ids
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._pack_latents
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline._unpack_latents
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    # Copied from diffusers.pipelines.flux.pipeline_flux.FluxPipeline.prepare_latents
    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        h_win,
        w_win,
        dtype,
        device,
        coords,
        latents=None,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        latent_coords = 2 * (coords // (self.vae_scale_factor * 2))
        h_win = 2 * (int(h_win) // (self.vae_scale_factor * 2))
        w_win = 2 * (int(w_win) // (self.vae_scale_factor * 2))
        max_h_start = max(0, latents.shape[-2] - h_win)
        max_w_start = max(0, latents.shape[-1] - w_win)
        latent_coords = latent_coords.copy()
        latent_coords[:, 0] = np.clip(latent_coords[:, 0], 0, max_h_start)
        latent_coords[:, 1] = np.clip(latent_coords[:, 1], 0, max_w_start)

        latents_batch = [latents[..., h_start : h_start + h_win, w_start : w_start + w_win] for h_start, w_start in latent_coords]
        latents_batch = torch.cat(latents_batch, dim=0)

        # HUY
        latents_batch = self._pack_latents(latents_batch, batch_size * len(latents_batch), num_channels_latents, h_win, w_win)
        latent_image_ids = self._prepare_latent_image_ids(batch_size, h_win // 2, w_win // 2, device, dtype)
        return latents_batch, latent_image_ids, latent_coords, h_win, w_win
    

    def prepare_latents_batch(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        h_win,
        w_win,
        h_stride,
        w_stride,
        dtype,
        device,
        generator,
        latent_coords,
        latents=None,
    ):
        assert latents is not None

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        h_win = 2 * (int(h_win) // (self.vae_scale_factor * 2))
        w_win = 2 * (int(w_win) // (self.vae_scale_factor * 2))
        max_h_start = max(0, latents.shape[-2] - h_win)
        max_w_start = max(0, latents.shape[-1] - w_win)
        latent_coords = latent_coords.copy()
        latent_coords[:, 0] = np.clip(latent_coords[:, 0], 0, max_h_start)
        latent_coords[:, 1] = np.clip(latent_coords[:, 1], 0, max_w_start)

        # create batch of patches
        latents_batch = [latents[..., h_start : h_start + h_win, w_start : w_start + w_win] for h_start, w_start in latent_coords]
        latents_batch = torch.cat(latents_batch, dim=0)

        latents_batch = self._pack_latents(latents_batch, batch_size*len(latents_batch), num_channels_latents, h_win, w_win)
        return latents_batch


    # Copied from diffusers.pipelines.controlnet_sd3.pipeline_stable_diffusion_3_controlnet.StableDiffusion3ControlNetPipeline.prepare_image
    def prepare_image(
        self,
        image,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        if isinstance(image, torch.Tensor):
            pass
        else:
            image = self.image_processor.preprocess(image, height=height, width=width)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        ) -> Tuple:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.scheduler.timesteps.device)
        step_index = self.scheduler.index_for_timestep(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        sigma = self.scheduler.sigmas[step_index]
        sigma_next = self.scheduler.sigmas[step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output

        # Cast sample back to model compatible dtype
        prev_sample = prev_sample.to(model_output.dtype)
        return (prev_sample,)

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        base_prompt_format: Optional[str] = None,
        combiner: Optional[str] = None,
        fall_back_prompt: Optional[str] = None,
        prompt_fragments: Optional[List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        h_win: Optional[int] = None,
        w_win: Optional[int] = None,
        h_stride: Optional[int] = None,
        w_stride: Optional[int] = None,
        # coords: Optional[np.array] = None,
        coords_list: Optional[List[np.array]] = None,
        label_list: Optional[List[np.array]] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 7.0,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        control_image: PipelineImageInput = None,
        label_map: PipelineImageInput = None,
        num_textures: Optional[int] = 3,
        control_mode: Optional[Union[int, List[int]]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        decode_once: bool = False,
        use_gaussian_mask: bool = False,
        debug_oom: bool = False,
        debug_size: Optional[int] = None,
        sigma: Optional[int] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        chunk_size: Optional[int] = None,
        out_dir: Optional[str] = None,
        anchor_points_for_viz: Optional[List[Tuple[int, int]]] = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            prompt_2 (`str` or `List[str]`, *optional*):
                The prompt or prompts to be sent to `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                will be used instead
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            control_image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.Tensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.Tensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be accepted
                as an image. The dimensions of the output image defaults to `image`'s dimensions. If height and/or
                width are passed, `image` is resized accordingly. If multiple ControlNets are specified in `init`,
                images must be passed as a list such that each element of the list can be correctly batched for input
                to a single ControlNet.
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            control_mode (`int` or `List[int]`,, *optional*, defaults to None):
                The control mode when applying ControlNet-Union.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.flux.FluxPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.flux.FluxPipelineOutput`] or `tuple`: [`~pipelines.flux.FluxPipelineOutput`] if `return_dict`
            is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the generated
            images.
        """

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(self.controlnet.nets) if isinstance(self.controlnet, FluxMultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = (
                mult * [control_guidance_start],
                mult * [control_guidance_end],
            )

        # 1. Check inputs. Raise error if not correct
        start_time = time.time()
        self.check_inputs(
            base_prompt_format,
            prompt_fragments,
            fall_back_prompt,
            height,
            width,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if fall_back_prompt is not None:
            batch_size = 1
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        dtype = self.transformer.dtype

        # 3. Prepare text embeddings
        start_time = time.time()
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        prompt_embeds_cache = {}
        
        if fall_back_prompt is None:
            raise ValueError("`fall_back_prompt` must be provided when using `prompt_fragments`.")
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=fall_back_prompt,
            prompt_2=None,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale
        )
        prompt_embeds_cache[fall_back_prompt] = (prompt_embeds, pooled_prompt_embeds, text_ids)

        # 4. Load or sample latents
        num_channels_latents = self.controlnet.config.in_channels // 4

        latent_paths = sorted(glob.glob(os.path.join(out_dir, f"latent_step*.npy")))
        if len(latent_paths) > 0:
            reload_step_idx = int(latent_paths[-1].split("latent_step")[-1].split(".")[0])  # resume from this step and onwards
            latents = torch.from_numpy(np.load(latent_paths[-1])).to(device, dtype=dtype)
        else:
            reload_step_idx = 0
            height_latent = 2 * (int(height) // (self.vae_scale_factor * 2))
            width_latent = 2 * (int(width) // (self.vae_scale_factor * 2))
            shape = (batch_size, num_channels_latents, height_latent, width_latent)
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)  # (1, 16, H//8, W//8)
            np.save(os.path.join(out_dir, f"latent_step{reload_step_idx:03d}.npy"), latents.detach().cpu().to(torch.float32).numpy())  # ensures same latents on different devices

        # c. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = 4096
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.16),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        if debug_oom:
            assert False, "not implemented"
            # coords = coords[:chunk_size]
            coords = coords[:10]
            chunk_size = 5
        
        # HUY
        step_to_viz_coords = {}
        if anchor_points_for_viz:
            for i in range(num_inference_steps):
                coords_this_step = set()
                for anchor_h, anchor_w in anchor_points_for_viz:
                    for h_start, w_start in coords_list[i]:
                        if (h_start <= anchor_h < h_start + h_win) and \
                           (w_start <= anchor_w < w_start + w_win):
                            coords_this_step.add((h_start, w_start))
                            break
                step_to_viz_coords[i] = coords_this_step

        # 5. Process the rest in chunks
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for step_i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                if step_i < reload_step_idx:
                    print(f"Skipping step {step_i+1}, already computed")
                    progress_bar.update()
                    continue

                # load checkpointed chunks
                chunk_dir = os.path.join(out_dir, f"latent_step{step_i+1:03d}")
                global_chunk_start = 0
                latents_canvas = None
                count_canvas = None
                if os.path.exists(chunk_dir):
                    latent_canvas_paths = sorted(glob.glob(os.path.join(chunk_dir, f"chunk_latents_canvas_*.npy")))
                    count_canvas_paths = sorted(glob.glob(os.path.join(chunk_dir, f"chunk_count_canvas_*.npy")))
                    n_latent_canvas = len(latent_canvas_paths)
                    n_count_canvas = len(count_canvas_paths)
                    n_joint = min(n_latent_canvas, n_count_canvas)
                    if n_joint > 0:
                        try:
                            latent_canvas_path = latent_canvas_paths[n_joint-1]
                            count_canvas_path = count_canvas_paths[n_joint-1]
                            global_chunk_start = int(latent_canvas_path.split("chunk_latents_canvas_start")[-1].split("_end")[1].split(".")[0])
                            latents_canvas = torch.from_numpy(np.load(latent_canvas_path)).to(device, dtype=dtype)
                            count_canvas = torch.from_numpy(np.load(count_canvas_path)).to(device, dtype=dtype)
                            print(f"[INFO] Step {step_i+1}, resumeing from chunk start {global_chunk_start}")
                        except:
                            print(f"[INFO] Current chunk corrupted, try previous chunk")
                            if n_joint > 1:
                                latent_canvas_path = latent_canvas_paths[n_joint-2]
                                count_canvas_path = count_canvas_paths[n_joint-2]
                                global_chunk_start = int(latent_canvas_path.split("chunk_latents_canvas_start")[-1].split("_end")[1].split(".")[0])
                                latents_canvas = torch.from_numpy(np.load(latent_canvas_path)).to(device, dtype=dtype)
                                count_canvas = torch.from_numpy(np.load(count_canvas_path)).to(device, dtype=dtype)
                                print(f"[INFO] Step {step_i+1}, resumeing from chunk start {global_chunk_start}")

                if latents_canvas is None:
                    latents_canvas = torch.zeros_like(latents)
                    count_canvas = torch.zeros_like(latents)
                    os.makedirs(chunk_dir, exist_ok=True)
                
                # HUY
                target_coords_set = step_to_viz_coords.get(step_i, set())

                print("after initializing image/count canvas, CPU", psutil.Process().memory_info().rss / 1e9, "GB, GPU", torch.cuda.memory_allocated() / 1e9, "GB")

                for chunk_i in tqdm(range(global_chunk_start, len(coords_list[step_i]), chunk_size), desc=f"Processing chunks"):
                    chunk_start = chunk_i
                    chunk_end = min(chunk_start + chunk_size, len(coords_list[step_i]))
                    coords_chunk = coords_list[step_i][chunk_start:chunk_end]

                    def prepare_control_latents(control_image):
                        control_latent_batch = []
                        for h_start, w_start in tqdm(coords_chunk, desc="Preparing control latent"):
                            control_image_pil = control_image.crop((w_start, h_start, w_start+w_win, h_start+h_win))
                            control_image_np = self.image_processor.pil_to_numpy(control_image_pil)
                            control_image_pt_cpu = self.image_processor.normalize(self.image_processor.numpy_to_pt(control_image_np))
                            control_image_pt_gpu = control_image_pt_cpu.to(device, dtype=dtype)
                            control_latent = retrieve_latents(self.vae.encode(control_image_pt_gpu), generator=generator)
                            control_latent = (control_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                            height_control_latent, width_control_latent = control_latent.shape[2:]
                            control_latent = self._pack_latents(
                                control_latent,
                                batch_size * num_images_per_prompt * control_latent.shape[0],
                                num_channels_latents,
                                height_control_latent,
                                width_control_latent,
                            ).to("cpu")
                            control_latent_batch.append(control_latent)
                            del control_image_pil, control_image_np, control_image_pt_cpu, control_image_pt_gpu, control_latent
                            torch.cuda.empty_cache()

                        # (1, 4096, 64)
                        return control_latent_batch

                    def prepare_one_hot(label_map):

                        label_map_one_hot_batch = []

                        latent_height = h_win // self.vae_scale_factor
                        latent_width = w_win // self.vae_scale_factor

                        for h_start, w_start in tqdm(coords_chunk, desc="Preparing label one-hot"):
                            label_map_pil = label_map.crop((w_start, h_start, w_start+w_win, h_start+h_win))
                            label_map_np = np.array(label_map_pil)
                            label_map_pt = torch.from_numpy(label_map_np).unsqueeze(0).unsqueeze(0).to(torch.long) # (1, 1, H, W)

                            # downsample label map to match dimensions (1, 1, 128, 128)
                            label_map_downsampled = torch.nn.functional.interpolate(
                                label_map_pt.float(),
                                size = (latent_height, latent_width),
                                mode='nearest'
                            ).long().squeeze(1).to(device)

                            # send label map image to one-hot
                            label_map_one_hot = torch.nn.functional.one_hot(
                                label_map_downsampled, 
                                num_classes=num_textures
                            ).permute(0, 3, 1, 2).to(torch.float16).cpu() # (B, C, H, W)
                            
                            label_map_one_hot_batch.append(label_map_one_hot)

                            del label_map_pil, label_map_np, label_map_pt, label_map_downsampled
                            torch.cuda.empty_cache()
                        
                        return label_map_one_hot_batch
                    

                    if type(control_image) == list:
                        control_latent_batch = [prepare_control_latents(control_image_) for control_image_ in control_image]
                    else:
                        control_latent_batch = prepare_control_latents(control_image)
                    
                    if type(label_map) == list:
                        label_map_one_hot_batch = [prepare_one_hot(label_map_) for label_map_ in label_map]
                    else:
                        label_map_one_hot_batch = prepare_one_hot(label_map)

                    print(f"Inference step {step_i+1} out of {num_inference_steps}, step 5a, part 1: {time.time() - start_time} seconds, CPU {psutil.Process().memory_info().rss / 1e9} GB, GPU {torch.cuda.memory_allocated() / 1e9} GB")

                    start_time = time.time()
                    controlnet_blocks_repeat = False

                    if type(control_image) == list:
                        if isinstance(control_mode, list) and len(control_mode) != len(control_image):
                            raise ValueError(
                                "For Multi-ControlNet, `control_mode` must be a list of the same "
                                + " length as the number of controlnets (control images) specified"
                            )
                        if not isinstance(control_mode, list):
                            control_mode = [control_mode] * len(control_image)
                        # set control mode
                        control_modes = []
                        for cmode in control_mode:
                            if cmode is None:
                                cmode = -1
                            control_mode = torch.tensor(cmode).view(-1, 1).expand(1, 1).to(device, dtype=torch.long)
                            control_modes.append(control_mode)
                        control_mode = control_modes
                    else:
                        # Here we ensure that `control_mode` has the same length as the control_image.
                        if control_mode is not None:
                            if not isinstance(control_mode, int):
                                raise ValueError(" For `FluxControlNet`, `control_mode` should be an `int` or `None`")
                            control_mode = torch.tensor(control_mode).to(device, dtype=torch.long)
                            control_mode = control_mode.view(-1, 1).expand(1, 1)

                    print(f"Inference step {step_i+1} out of {num_inference_steps}, step 5a, part 2: {time.time() - start_time} seconds, CPU {psutil.Process().memory_info().rss / 1e9} GB, GPU {torch.cuda.memory_allocated() / 1e9} GB")

                    # b. Process latents into windows

                    latents_batch, latent_image_ids, latent_coords, latent_h_win, latent_w_win = self.prepare_latents(  # (1,4096,64)
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        h_win,
                        w_win,
                        dtype,
                        device,
                        coords_chunk,
                        latents,
                    )
                    print(f"Inference step {step_i+1} out of {num_inference_steps}, step 5b: {time.time() - start_time} seconds, CPU {psutil.Process().memory_info().rss / 1e9} GB, GPU {torch.cuda.memory_allocated() / 1e9} GB")

                    # d. Create tensor stating which controlnets to keep
                    controlnet_keep = []
                    for i in range(len(timesteps)):
                        keeps = [
                            1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                            for s, e in zip(control_guidance_start, control_guidance_end)
                        ]
                        controlnet_keep.append(keeps[0] if self.controlnet.__class__.__name__ == "FluxControlNetModel" else keeps)
                    
                    use_guidance = self.controlnet.nets[0].config.guidance_embeds if type(control_image) == list else self.controlnet.config.guidance_embeds
                    if isinstance(controlnet_keep[step_i], list):
                        cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[step_i])]
                    else:
                        controlnet_cond_scale = controlnet_conditioning_scale
                        if isinstance(controlnet_cond_scale, list):
                            controlnet_cond_scale = controlnet_cond_scale[0]
                        cond_scale = controlnet_cond_scale * controlnet_keep[step_i]

                    for loop_idx in tqdm(range(len(latents_batch)), desc="Processing multidiffusion windows"):
                        
                        # get current window and crop label map
                        h_start_win, w_start_win = coords_chunk[loop_idx]
                        label_map_pil_crop = label_map.crop((w_start_win, h_start_win, w_start_win + w_win, h_start_win + h_win))
                        label_map_np_crop = np.array(label_map_pil_crop)

                        # check dominant encoding in label map to determine prompt
                        total_pixels = np.count_nonzero(label_map_np_crop)
                        labels_present = []

                        if total_pixels > 0:
                            for i in range(1, num_textures):
                                label_index = i
                                num_pixels_for_texture = np.count_nonzero(label_map_np_crop == label_index)

                                if (num_pixels_for_texture / total_pixels) > 0.10:
                                    labels_present.append(label_index)
                        
                        if len(labels_present) == 0:
                            selected_prompt = fall_back_prompt
                        else:
                            fragments = [prompt_fragments[i - 1] for i in labels_present]

                            seperator = f" {combiner} "
                            combined_fragments = seperator.join(fragments)

                            selected_prompt = base_prompt_format.format(combined_fragments)
                        
                        if selected_prompt not in prompt_embeds_cache:
                            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                                prompt=selected_prompt,
                                prompt_2=None,
                                device=device,
                                num_images_per_prompt=num_images_per_prompt,
                                max_sequence_length=max_sequence_length,
                                lora_scale=lora_scale
                            )
                            prompt_embeds_cache[selected_prompt] = (prompt_embeds, pooled_prompt_embeds, text_ids)

                        (
                            prompt_embeds_window,
                            pooled_prompt_embeds_window,
                            text_ids_window,
                        ) = prompt_embeds_cache[selected_prompt]

                        latents_win = latents_batch[loop_idx:loop_idx+1]
                        if type(control_image) == list:
                            control_image_win = [control_latent_batch_[loop_idx].clone().to(device, dtype=latents_win.dtype) for control_latent_batch_ in control_latent_batch]
                        else:
                            control_image_win = control_latent_batch[loop_idx].clone()
                            control_image_win = control_image_win.to(device, dtype=latents_win.dtype)

                        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                        timestep = t.expand(latents_win.shape[0]).to(latents.dtype)
                        guidance = torch.tensor([guidance_scale], device=device) if use_guidance else None
                        guidance = guidance.expand(latents.shape[0]) if guidance is not None else None
                        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
                            hidden_states=latents_win,
                            controlnet_cond=control_image_win,
                            controlnet_mode=control_mode,
                            conditioning_scale=cond_scale,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds_window,
                            encoder_hidden_states=prompt_embeds_window,
                            txt_ids=text_ids_window,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )

                        if isinstance(label_map_one_hot_batch, list) and label_map_one_hot_batch and isinstance(label_map_one_hot_batch[0], list):
                            label_one_hot_src = label_map_one_hot_batch[0]
                        else:
                            label_one_hot_src = label_map_one_hot_batch

                        # get current window of label map one hot
                        label_one_hot_win = label_one_hot_src[loop_idx].clone().to(device, dtype=latents_win.dtype)

                        # unpack latent window to get (1, 16, 128, 128)
                        unpacked_latents_win = self._unpack_latents(
                            latents_win,
                            h_win,
                            w_win,
                            self.vae_scale_factor
                        )
                        
                        # concatenate along channels to get (1, 19, 128, 128)
                        latents_win_unpacked_for_transformer = torch.cat([unpacked_latents_win, label_one_hot_win], dim=1)
                        latent_height = latents_win_unpacked_for_transformer.shape[2]
                        latent_width = latents_win_unpacked_for_transformer.shape[3] 

                        # pack to send to transformer
                        latents_win_for_transformer = self._pack_latents(
                            latents_win_unpacked_for_transformer,
                            latents_win_unpacked_for_transformer.shape[0], 
                            latents_win_unpacked_for_transformer.shape[1], 
                            latent_height, 
                            latent_width
                        )

                        noise_pred = self.transformer(
                            hidden_states=latents_win_for_transformer,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            pooled_projections=pooled_prompt_embeds_window,
                            encoder_hidden_states=prompt_embeds_window,
                            controlnet_block_samples=controlnet_block_samples,
                            controlnet_single_block_samples=controlnet_single_block_samples,
                            txt_ids=text_ids_window,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            controlnet_blocks_repeat=controlnet_blocks_repeat,
                        )[0]
                        if torch.isnan(noise_pred).any():
                            import pdb; pdb.set_trace()
                        
                        current_coords = tuple(coords_chunk[loop_idx])
                        # huy's code trial run
                        if current_coords in target_coords_set:
                            sigma = self.scheduler.sigmas[self.scheduler.index_for_timestep(t)]

                            approx_clean_packed = (latents_win - sigma * noise_pred) / (1.0 - sigma)

                            approx_clean_unpacked = self._unpack_latents(approx_clean_packed, h_win, w_win, self.vae_scale_factor)
                            approx_clean_scaled = (approx_clean_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                            decoded_image = self.vae.decode(approx_clean_scaled, return_dict=False)[0]
                            decoded_image_processed = self.image_processor.postprocess(decoded_image, output_type=output_type)[0]
                            decoded_image_processed.save(os.path.join(out_dir, f"step_{step_i}_patch{loop_idx}.png"))

                        latents_win = self.step(noise_pred, t, latents_win)[0]
                        latent_h_start_, latent_w_start_ = latent_coords[loop_idx]
                        latents_canvas[..., latent_h_start_:latent_h_start_ + latent_h_win, latent_w_start_:latent_w_start_ + latent_w_win] += self._unpack_latents(latents_win, h_win, w_win, self.vae_scale_factor)
                        count_canvas[..., latent_h_start_:latent_h_start_ + latent_h_win, latent_w_start_:latent_w_start_ + latent_w_win] += 1

                        del control_image_win, noise_pred, controlnet_block_samples, controlnet_single_block_samples
                        torch.cuda.empty_cache()
                
                    del latents_batch
                    torch.cuda.empty_cache()

                    np.save(os.path.join(chunk_dir, f"chunk_latents_canvas_start{chunk_start:08d}_end{chunk_end:08d}.npy"), latents_canvas.detach().cpu().to(torch.float32).numpy())
                    np.save(os.path.join(chunk_dir, f"chunk_count_canvas_start{chunk_start:08d}_end{chunk_end:08d}.npy"), count_canvas.detach().cpu().to(torch.float32).numpy())
                
                latents = latents_canvas / count_canvas

                del latents_canvas, count_canvas
                torch.cuda.empty_cache()

                latents_dtype = latents.dtype
                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if step_i == len(timesteps) - 1 or ((step_i + 1) > num_warmup_steps and (step_i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

                np.save(os.path.join(out_dir, f"latent_step{step_i+1:03d}.npy"), latents.detach().cpu().to(torch.float32).numpy())
                prev_path = os.path.join(out_dir, f"latent_step{step_i:03d}.npy")
                if os.path.exists(prev_path):
                    os.system(f"rm -rf {prev_path}")
                os.system(f"rm -rf {chunk_dir}")

        if output_type == "latent":
            image = latents

        else:
            intermediate_list = []
            if decode_once:
                latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                image = self.vae.decode(latents, return_dict=False)[0]
                image = self.image_processor.postprocess(image, output_type=output_type)
            else:
                from PIL import Image
                np2pil = lambda x: Image.fromarray(np.round(np.transpose(np.clip(x * 0.5 + 0.5, 0, 1), (0, 2, 3, 1))[0] * 255).astype("uint8"))

                # create square mask
                def square_blur(h, w, square_min=0.001):
                    y, x = np.ogrid[:h, :w]
                    cy, cx = h // 2, w // 2
                    dist = np.maximum(np.abs(y - cy), np.abs(x - cx))
                    weight = 1 - (dist / dist.max()) * (1 - square_min)
                    return weight

                weight = square_blur(h_win, w_win)
                weight = np.expand_dims(np.stack([weight, weight, weight], axis=0), axis=0)

                # decode patch by patch then blend (OOM when > 4096x4096)
                bs, _, latent_h, latent_w = latents.shape
                shape = (bs, 3, latent_h*8, latent_w*8)
                print("before initializing image/count canvas", psutil.Process().memory_info().rss / 1e9, "GB")
                tmp_dir = os.path.join(out_dir, "tmp")
                os.makedirs(tmp_dir, exist_ok=True)
                image_canvas = np.memmap(os.path.join(tmp_dir, f"image_canvas.npy"), mode='w+', shape=shape, dtype='float32')
                count_canvas = np.memmap(os.path.join(tmp_dir, f"count_canvas.npy"), mode='w+', shape=shape, dtype='float32')
                print("after initializing image/count canvas", psutil.Process().memory_info().rss / 1e9, "GB")
                # a40, 1024px, 36159/46068MB
                for chunk_i in tqdm(range(0, len(coords_list[0]), chunk_size), desc=f"Processing chunks"):
                    chunk_start = chunk_i
                    chunk_end = min(chunk_start + chunk_size, len(coords_list[0]))
                    coords_chunk = coords_list[0][chunk_start:chunk_end]
                    latents_batch, latent_image_ids, latent_coords, latent_h_win, latent_w_win = self.prepare_latents(  # (1,4096,64)
                        batch_size * num_images_per_prompt,
                        num_channels_latents,
                        h_win,
                        w_win,
                        dtype,
                        device,
                        coords_chunk,
                        latents,
                    )
                    latents_batch = (latents_batch / self.vae.config.scaling_factor) + self.vae.config.shift_factor
                    latents_batch = self._unpack_latents(latents_batch, h_win, w_win, self.vae_scale_factor)
                    count = 0
                    for i, (h_start_, w_start_) in enumerate(tqdm(coords_chunk, desc="Decoding patches")):
                        image_canvas[..., h_start_:h_start_ + h_win, w_start_:w_start_ + w_win] += self.vae.decode(latents_batch[i:i+1], return_dict=False)[0].to('cpu', dtype=torch.float32).numpy() * weight
                        count_canvas[..., h_start_:h_start_ + h_win, w_start_:w_start_ + w_win] += weight
                        if count < 5:
                            if label_list[0][i] == 1:
                                np2pil(self.vae.decode(latents_batch[i:i+1], return_dict=False)[0].to('cpu', dtype=torch.float32).numpy()).save(os.path.join(out_dir, f"chunk{chunk_i:03d}_patch{i:03d}_hstart{h_start_}_wstart{w_start_}.png"))
                                count += 1
                    del latents_batch
                    torch.cuda.empty_cache()

                print("after decoding patches, CPU", psutil.Process().memory_info().rss / 1e9, "GB, GPU", torch.cuda.memory_allocated() / 1e9, "GB")

                # combine with bicubic upsampled image
                for i, (h_start_, w_start_) in enumerate(tqdm(coords_list[0], desc="Combining generated patches with bicubic upsampled image")):
                    if label_list[0][i] == 1:
                        im_crop_torch = image_canvas[..., h_start_:h_start_+h_win, w_start_:w_start_+w_win] / count_canvas[..., h_start_:h_start_+h_win, w_start_:w_start_+w_win]
                        im_crop = np2pil(im_crop_torch)
                        if type(control_image) == list:
                            control_image[0].paste(im_crop, (w_start_, h_start_))
                        else:
                            control_image.paste(im_crop, (w_start_, h_start_))
                        del im_crop_torch, im_crop
                        torch.cuda.empty_cache()
                print("after combining, CPU", psutil.Process().memory_info().rss / 1e9, "GB, GPU", torch.cuda.memory_allocated() / 1e9, "GB")
                del image_canvas, count_canvas
        
        # Offload all models
        self.maybe_free_model_hooks()

        # remove mem files
        os.system(f"rm -rf {tmp_dir}")

        if not return_dict:
            if type(control_image) == list:
                return (control_image[0], intermediate_list)
            else:
                return (control_image, intermediate_list)

        return FluxPipelineOutput(images=image)


# version 0: 
#  - memory-monitored and reduced scripts, deleted variables after use
#  - passed in coords instead of creating coords inside the function
# 
# version 1: 
#  - save and reload latents across timesteps
# 
# version 2: 
#  - scaling up to even bigger inference region: reducing even more cpu/gpu memory usage
#
# version 3: 
#  - experiment with different muldiffusion window strategies
#     - requires all windows to be processed for the same number of steps (boundary windows tricky)
#     - naive #1: upscale the entire valid region crop, instead of a finegrained, patchified valid region;
#                 randomize stride within this cropped region
#
# version 4: 
#  - better color blending
#     - gaussian mask
#
# version 5: 
#  - get inference_05 to not OOM
#
# version 6: 
#  - save/reload chunks
#
# version 7: 
#  - randomize stride within this cropped region
