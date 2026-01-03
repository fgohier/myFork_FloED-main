import imageio
import inspect
from typing import Callable, List, Optional, Union
from dataclasses import dataclass
import time
import numpy as np
import torch
from tqdm import tqdm
import os
import cv2
from PIL import Image
from natsort import natsorted
from torchvision import transforms
from diffusers.utils import is_accelerate_available
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate, logging, BaseOutput

from einops import rearrange

from ..models.unet import UNet3DConditionModel
from ..models.sparse_controlnet import SparseControlNetModel
from datetime import datetime
import torch.nn.functional as F

logger = logging.get_logger(__name__)
from Flow.flow_comp_raft import RAFT_bi
from Flow.RAFT.utils.flow_viz_pt import flow_to_image


def read_frame_from_videos_mp4(vname):
    frames = []
    vidcap = cv2.VideoCapture(vname)
    success, image = vidcap.read()
    count = 0
    while success:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    return frames


def read_frame_from_videos(vname):
    frames = []
    lst = os.listdir(vname)
    lst = natsorted(lst)
    fr_lst = [vname + '/' + name for name in lst]
    for fr in fr_lst:
        image = Image.open(fr).convert('RGB')
        frames.append(image)
    return frames


class Stack(object):
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        mode = img_group[0].mode
        if mode == '1':
            img_group = [img.convert('L') for img in img_group]
            mode = 'L'
        if mode == 'L':
            return np.stack([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif mode == 'RGB':
            if self.roll:
                return np.stack([np.array(x)[:, :, ::-1] for x in img_group],
                                axis=2)
            else:
                return np.stack(img_group, axis=2)
        else:
            raise NotImplementedError(f"Image mode {mode}")


class ToTorchFormatTensor(object):
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 3, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(
                pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        img = img.float().div(255) if self.div else img.float()
        return img


def to_tensors():
    return transforms.Compose([Stack(), ToTorchFormatTensor()])


def read_mask(mpath, size):
    masks = []
    mnames = os.listdir(mpath)
    mnames = natsorted(mnames)
    for mp in mnames:
        m = Image.open(os.path.join(mpath, mp))
        m = m.resize(size, Image.NEAREST)
        m = np.array(m.convert('L'))
        m = np.array(m > 0).astype(np.uint8)
        m = cv2.dilate(m,
                       cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
                       iterations=5)
        masks.append(Image.fromarray(m * 255))
    return masks


@dataclass
class AnimationPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class AnimationPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
            self,
            vae: AutoencoderKL,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: UNet3DConditionModel,
            Flow_estimator: RAFT_bi,
            scheduler: Union[
                DDIMScheduler,
                PNDMScheduler,
                LMSDiscreteScheduler,
                EulerDiscreteScheduler,
                EulerAncestralDiscreteScheduler,
                DPMSolverMultistepScheduler,
            ],
            controlnet: Union[SparseControlNetModel, None] = None,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)
        self.Flow_estimator = Flow_estimator
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        if is_accelerate_available():
            from accelerate import cpu_offload
        else:
            raise ImportError("Please install accelerate via `pip install accelerate`")

        device = torch.device(f"cuda:{gpu_id}")

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            if cpu_offloaded_model is not None:
                cpu_offload(cpu_offloaded_model, device)

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1: -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        text_embeddings = text_embeddings[0]

        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            uncond_embeddings = uncond_embeddings[0]

            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    def decode_latents(self, latents, frames, masks):
        video_length = latents.shape[2]
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        video = []
        for frame_idx in tqdm(range(latents.shape[0])):
            video.append(self.vae.decode(latents[frame_idx:frame_idx + 1]).sample)
        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b c f h w", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        frames = (frames / 2 + 0.5).clamp(0, 1)

        video = rearrange(video, "b c t h w -> t b c h w")
        masks = rearrange(masks, "b c t h w -> t b c h w")
        frames = rearrange(frames, "b c t h w -> t b c h w")

        video_blur = []
        for frame, mask, init_frame in zip(video, masks, frames):
            frame = frame.squeeze(0).transpose(0, 1).transpose(1, 2)
            mask = mask.squeeze(0).transpose(0, 1).transpose(1, 2)
            init_frame = init_frame.squeeze(0).transpose(0, 1).transpose(1, 2)

            frame = (frame * 255).cpu().float().numpy().astype(np.uint8)
            mask = mask.cpu().float().numpy().astype(np.uint8)
            init_frame = (init_frame * 255).cpu().float().numpy().astype(np.uint8)
            frame = np.array(frame)
            mask = np.array(mask)
            init_frame = np.array(init_frame)

            mask_blurred = cv2.GaussianBlur(mask * 255, (21, 21), 0) / 255
            mask_blurred = mask_blurred[:, :, np.newaxis]
            mask = 1 - (1 - mask) * (1 - mask_blurred)

            image_pasted = init_frame * (1 - mask) + frame * mask
            image_pasted = image_pasted.astype(frame.dtype)
            image = Image.fromarray(image_pasted)

            video_blur.append(image)
        video = to_tensors()(video_blur)
        video = video.unsqueeze(0).transpose(1, 2)
        video_m = video[:, :, 1:, :, :]
        video_R = video_m.cpu().float().numpy()

        video = (video_m * 255).byte()
        video = video[0].cpu().numpy()
        video = np.transpose(video, (1, 2, 3, 0))

        num_frames, height, width, channels = video.shape
        if channels != 3:
            raise ValueError("Only 3-channel (RGB) videos are supported.")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("output_video.mp4", fourcc, 15, (width, height))

        for frame in video:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        out.release()
        return video_R

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, prompt, height, width, callback_steps):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
                callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, batch_size, num_channels_latents, video_length, height, width, dtype, device, generator,
                        latents=None):
        shape = (
            batch_size, num_channels_latents, video_length, height // self.vae_scale_factor,
            width // self.vae_scale_factor)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            rand_device = "cpu" if device.type == "mps" else device

            if isinstance(generator, list):
                shape = shape
                latents = [
                    torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype)
                    for i in range(batch_size)
                ]
                latents = torch.cat(latents, dim=0).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def save_tensor_as_image(self, tensor, filename):
        if tensor.ndim == 4:
            tensor = tensor[0]

        tensor = tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        image = Image.fromarray(tensor)
        image.save(filename)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            video_length: Optional[int],
            frames_path: str,
            masks_path: str,
            first_frame_path: str,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_videos_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "tensor",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            controlnet_images: torch.FloatTensor = None,
            controlnet_image_index: list = [0],
            controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
            **kwargs,
    ):
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        self.check_inputs(prompt, height, width, callback_steps)

        batch_size = 1
        if latents is not None:
            batch_size = latents.shape[0]
        if isinstance(prompt, list):
            batch_size = len(prompt)

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale > 1.0

        text_embeddings = self._encode_prompt(
            prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt
        )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            video_length,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )
        
        frames = read_frame_from_videos(frames_path)
        frames = [f.resize((width, height)) for f in frames]
        frames = to_tensors()(frames)
        frames = frames[:16, :, :, :]
        
        mask = read_mask(masks_path, (width, height))
        mask = to_tensors()(mask)
        mask = mask[:16, :, :, :]
        frames = frames.half()
        mask = mask.half()
        
        torch.cuda.empty_cache()
        frames = frames * 2 - 1
        frames, mask = frames.to(self.vae.device), mask.to(self.vae.device)
        
        masked_video = frames * (1 - mask)
        mask_pixel = mask
        mask_pixel_copy = mask.transpose(0, 1).unsqueeze(0)
        masked_video_copy = masked_video.transpose(0, 1).unsqueeze(0)
        frame_copy = frames.transpose(0, 1).unsqueeze(0)
        gt_optical_flow = self.Flow_estimator(frames.unsqueeze(0))
        corrupted_optical_flow = self.Flow_estimator(masked_video.unsqueeze(0))

        frames = frames.transpose(0, 1).unsqueeze(0)
        frames = rearrange(frames, "b c f h w -> (b f) c h w")
        frames_latent = self.vae.encode(frames).latent_dist
        frames_latent = frames_latent.sample()
        frames_latent = rearrange(frames_latent, "(b f) c h w -> b c f h w", f=video_length)
        frames_latent = frames_latent[:, :, 0, :, :].detach()
        latent_first = frames_latent.unsqueeze(2)

        image = Image.open(first_frame_path).convert('RGB')
        image = image.resize((width, height), Image.NEAREST)
        image = [image]
        image = to_tensors()(image)
        image = image * 2 - 1
        image = image.half()
        image = image.to(latents.device)
        image = self.vae.encode(image).latent_dist
        image = image.sample()
        image = image.unsqueeze(2)
        
        torch.cuda.empty_cache()

        masked_video = masked_video.transpose(0, 1).unsqueeze(0)
        masked_video = rearrange(masked_video, "b c f h w -> (b f) c h w")
        masked_video = self.vae.encode(masked_video).latent_dist
        masked_video = masked_video.sample()
        masked_video = rearrange(masked_video, "(b f) c h w -> b c f h w", f=video_length)
        
        mask = mask * 2 - 1
        mask = mask.unsqueeze(0)
        mask = rearrange(mask, "b f c h w -> (b f) c h w")
        mask = torch.nn.functional.interpolate(mask, size=masked_video.shape[-2:])
        mask = rearrange(mask, "(b f) c h w -> b c f h w", f=video_length)

        latents_dtype = latents.dtype
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        timesteps_durations = []
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                torch.cuda.synchronize()
                start_time = time.time()
                
                latents = torch.cat([image, latents[:, :, 1:, :, :]], dim=2)
                latents_input = torch.concat([latents, masked_video, mask], dim=1)

                latent_model_input = torch.cat([latents_input] * 2) if do_classifier_free_guidance else latents_input
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                down_block_additional_residuals = mid_block_additional_residual = None
                if (getattr(self, "controlnet", None) != None) and (controlnet_images != None):
                    assert controlnet_images.dim() == 5

                    controlnet_noisy_latents = latent_model_input
                    controlnet_prompt_embeds = text_embeddings

                    controlnet_images = controlnet_images.to(latents.device)

                    controlnet_cond_shape = list(controlnet_images.shape)
                    controlnet_cond_shape[2] = video_length
                    controlnet_cond = torch.zeros(controlnet_cond_shape).to(latents.device)

                    controlnet_conditioning_mask_shape = list(controlnet_cond.shape)
                    controlnet_conditioning_mask_shape[1] = 1
                    controlnet_conditioning_mask = torch.zeros(controlnet_conditioning_mask_shape).to(latents.device)

                    assert controlnet_images.shape[2] >= len(controlnet_image_index)
                    controlnet_cond[:, :, controlnet_image_index] = controlnet_images[:, :, :len(controlnet_image_index)]
                    controlnet_conditioning_mask[:, :, controlnet_image_index] = 1

                    down_block_additional_residuals, mid_block_additional_residual = self.controlnet(
                        controlnet_noisy_latents, t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_mask=controlnet_conditioning_mask,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=False, return_dict=False,
                    )

                if i == 0:
                    noise_pred, predict_optical_flow = self.unet(
                        latent_model_input, t,
                        encoder_hidden_states=text_embeddings,
                        corrupted_optical_flow=corrupted_optical_flow,
                        label=False,
                        tag='test',
                        down_block_additional_residuals=down_block_additional_residuals,
                        mid_block_additional_residual=mid_block_additional_residual,
                    )
                else:
                    noise_pred, predict_optical_flow = self.unet(
                        latent_model_input, t,
                        encoder_hidden_states=text_embeddings,
                        corrupted_optical_flow=corrupted_optical_flow,
                        label=True,
                        down_block_additional_residuals=down_block_additional_residuals,
                        mid_block_additional_residual=mid_block_additional_residual,
                    )
                noise_pred = noise_pred.sample.to(dtype=latents_dtype)
                predict_optical_flow = predict_optical_flow.to(dtype=latents_dtype)

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
                torch.cuda.synchronize()
                end_time = time.time()
                duration = end_time - start_time
                timesteps_durations.append(duration)
                
        average_time = sum(timesteps_durations) / len(timesteps_durations)
        print(f'Average time per iteration: {average_time:.4f} seconds')


        video = self.decode_latents(latents, frame_copy, mask_pixel_copy)

        if output_type == "tensor":
            video = torch.from_numpy(video)

        if not return_dict:
            return video

        return AnimationPipelineOutput(videos=video)