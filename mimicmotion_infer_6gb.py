#!/usr/bin/env python
"""MimicMotion inference script for serverless deployment with tunable quality."""

from __future__ import annotations

import argparse
import inspect
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_pil_image

from diffusers.utils.torch_utils import is_compiled_module

from mimicmotion.utils.geglu_patch import patch_geglu_inplace
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.dwpose.preprocess import get_image_pose, get_video_pose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MimicMotion inference for serverless GPUs.")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path or HF model id for SVD base model.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to MimicMotion checkpoint (.pth).")
    parser.add_argument("--ref_image_path", type=str, required=True, help="Reference identity image.")
    parser.add_argument("--driving_video_path", type=str, required=True, help="Driving video for motion/pose.")
    parser.add_argument("--output_video_path", type=str, default="outputs/mimicmotion_out.mp4")
    parser.add_argument("--pose_preview_path", type=str, default="outputs/dwpose_preview.mp4")
    parser.add_argument("--height", type=int, default=1024, help="Output height (default: 1024).")
    parser.add_argument("--width", type=int, default=576, help="Output width (default: 576).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--sample_stride", type=int, default=1, help="Frame sampling stride for DWPose.")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--steps", type=int, default=35, help="Diffusion inference steps.")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--noise_aug_strength", type=float, default=0.0)
    parser.add_argument("--tile_size", type=int, default=32, help="Temporal tile size for denoising.")
    parser.add_argument("--tile_overlap", type=int, default=8, help="Temporal overlap for denoising tiles.")
    parser.add_argument(
        "--vae_decode_chunk_size",
        type=int,
        default=8,
        help="VAE decode chunk size. Increase on strong GPUs for faster decoding.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_fp8", action="store_true", help="Disable FP8 UNet quantization.")
    parser.add_argument(
        "--prefer_gpu_video_codec",
        action="store_true",
        default=True,
        help="Prefer NVENC for output encoding when ffmpeg supports it.",
    )
    parser.add_argument("--no_gpu_video_codec", dest="prefer_gpu_video_codec", action="store_false")
    parser.add_argument(
        "--keep_input_audio",
        action="store_true",
        default=True,
        help="Attach audio track from driving video to generated output when available.",
    )
    parser.add_argument("--no_input_audio", dest="keep_input_audio", action="store_false")
    return parser.parse_args()


def resize_ref_image(ref_image_path: str, width: int, height: int) -> np.ndarray:
    image = Image.open(ref_image_path).convert("RGB")
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return np.asarray(image, dtype=np.uint8)


def _write_video_ffmpeg_nvenc(frames_hwc: np.ndarray, output_path: str, fps: int) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    height, width = frames_hwc.shape[1], frames_hwc.shape[2]
    cmd = [
        ffmpeg_bin,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-preset",
        "p5",
        "-cq",
        "19",
        "-pix_fmt",
        "yuv420p",
        output_path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.stdin.write(frames_hwc.tobytes())
        proc.stdin.close()
        ret = proc.wait()
        return ret == 0 and Path(output_path).exists() and Path(output_path).stat().st_size > 0
    except Exception:
        return False


def write_video(frames: np.ndarray, output_path: str, fps: int, prefer_gpu_video_codec: bool = True) -> None:
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D frames, got {frames.shape}")
    if frames.shape[-1] == 3:
        frames_hwc = frames
    elif frames.shape[1] == 3:
        frames_hwc = np.transpose(frames, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected frames with channels=3, got {frames.shape}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if prefer_gpu_video_codec:
        if _write_video_ffmpeg_nvenc(frames_hwc, output.as_posix(), fps):
            return
    try:
        imageio.mimwrite(
            output.as_posix(),
            frames_hwc,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )
    except Exception:
        # Fallback to OpenCV if ffmpeg/imageio backend is unavailable.
        height, width = frames_hwc.shape[1], frames_hwc.shape[2]
        writer = cv2.VideoWriter(output.as_posix(), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in frames_hwc:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()


def attach_audio_from_driving_video(output_video_path: str, driving_video_path: str) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        print("ffmpeg not found. Skipping audio mux.")
        return False

    output_path = Path(output_video_path)
    temp_muxed = output_path.with_suffix(".with_audio.mp4")
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(output_path),
        "-i",
        driving_video_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(temp_muxed),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"Audio mux failed, keeping silent output. Reason: {exc}")
        return False

    if not temp_muxed.exists() or temp_muxed.stat().st_size <= 0:
        return False
    temp_muxed.replace(output_path)
    return True


def preprocess_dwpose(
    driving_video_path: str,
    ref_image_path: str,
    width: int,
    height: int,
    sample_stride: int,
    pose_preview_path: str,
    fps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ref_image_np = resize_ref_image(ref_image_path, width, height)

    # Extract DWPose from reference image + driving video before generation.
    image_pose = get_image_pose(ref_image_np)
    video_pose = get_video_pose(driving_video_path, ref_image_np, sample_stride=sample_stride)

    if video_pose.shape[-2:] != (height, width):
        resized_pose = [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR) for frame in video_pose
        ]
        video_pose = np.stack(resized_pose, axis=0)

    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose], axis=0)
    write_video(pose_pixels.astype(np.uint8), pose_preview_path, fps=fps, prefer_gpu_video_codec=False)

    pose_tensor = torch.from_numpy(pose_pixels.copy()).float() / 127.5 - 1.0
    image_tensor = torch.from_numpy(ref_image_np[None].copy()).permute(0, 3, 1, 2).float() / 127.5 - 1.0
    return pose_tensor, image_tensor


def apply_low_vram_mode(pipeline, device: torch.device) -> None:
    target_dtype = torch.float16 if device.type == "cuda" else torch.float32
    for module_name in ("unet", "vae", "image_encoder", "pose_net"):
        module = getattr(pipeline, module_name, None)
        if module is not None and hasattr(module, "to"):
            # Required Low-VRAM mode: fp16 module weights on target device.
            module.to(device, dtype=target_dtype)
    # Keep attention slicing for memory safety on constrained GPUs.
    if hasattr(pipeline, "enable_attention_slicing"):
        pipeline.enable_attention_slicing(1)


def apply_unet_fp8_quantization(unet) -> None:
    try:
        from optimum.quanto import freeze, qfloat8, quantize

        quantize(unet, weights=qfloat8)
        freeze(unet)
        print("UNet FP8 quantization: enabled (optimum-quanto qfloat8).")
    except Exception as exc:
        print(
            "UNet FP8 quantization was requested but could not be applied "
            f"(reason: {exc}). Continuing in fp16."
        )


def patch_tiled_vae_decode(pipeline, default_chunk_size: int = 2) -> None:
    chunk_size = max(1, int(default_chunk_size))

    def decode_latents_tiled(latents: torch.Tensor, num_frames: int, decode_chunk_size: int = chunk_size):
        effective_chunk = max(1, int(decode_chunk_size))
        latents_2d = latents.flatten(0, 1)
        latents_2d = latents_2d / pipeline.vae.config.scaling_factor

        forward_vae_fn = pipeline.vae._orig_mod.forward if is_compiled_module(pipeline.vae) else pipeline.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        chunks = []
        for idx in range(0, latents_2d.shape[0], effective_chunk):
            latent_chunk = latents_2d[idx : idx + effective_chunk]
            decode_kwargs = {"num_frames": latent_chunk.shape[0]} if accepts_num_frames else {}
            decoded_chunk = pipeline.vae.decode(latent_chunk, **decode_kwargs).sample
            chunks.append(decoded_chunk.cpu())
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        frames = torch.cat(chunks, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4).float()
        return frames

    pipeline.decode_latents = decode_latents_tiled


def run_inference(args: argparse.Namespace) -> dict:
    if args.batch_size != 1:
        raise ValueError("MimicMotion pipeline currently expects batch_size=1.")
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("height/width must be divisible by 8.")

    patch_geglu_inplace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    infer_dtype = torch.float16 if device.type == "cuda" else torch.float32
    if device.type == "cpu":
        raise RuntimeError("CUDA GPU is required for MimicMotion inference. CPU mode is not supported.")

    infer_cfg = SimpleNamespace(base_model_path=args.base_model_path, ckpt_path=args.ckpt_path)
    pipeline = create_pipeline(infer_cfg, device)

    apply_low_vram_mode(pipeline, device)
    if not args.disable_fp8:
        apply_unet_fp8_quantization(pipeline.unet)
    patch_tiled_vae_decode(pipeline, default_chunk_size=args.vae_decode_chunk_size)

    pose_pixels, image_pixels = preprocess_dwpose(
        driving_video_path=args.driving_video_path,
        ref_image_path=args.ref_image_path,
        width=args.width,
        height=args.height,
        sample_stride=args.sample_stride,
        pose_preview_path=args.pose_preview_path,
        fps=args.fps,
    )

    ref_images = [to_pil_image(img.to(torch.uint8)) for img in ((image_pixels + 1.0) * 127.5)]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    with torch.inference_mode():
        output = pipeline(
            ref_images,
            image_pose=pose_pixels.to(device=device, dtype=infer_dtype),
            num_frames=pose_pixels.size(0),
            tile_size=min(args.tile_size, pose_pixels.size(0)),
            tile_overlap=args.tile_overlap,
            height=args.height,
            width=args.width,
            fps=args.fps,
            noise_aug_strength=args.noise_aug_strength,
            num_inference_steps=args.steps,
            generator=generator,
            min_guidance_scale=args.guidance_scale,
            max_guidance_scale=args.guidance_scale,
            decode_chunk_size=args.vae_decode_chunk_size,
            num_videos_per_prompt=1,
            output_type="pt",
            device=device,
        )

    frames = output.frames
    if frames.ndim == 5:
        frames = frames[0]
    if frames.shape[-1] == 3:
        frames = (frames.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    elif frames.shape[1] == 3:
        frames = (frames.permute(0, 2, 3, 1).clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    else:
        raise RuntimeError(f"Unexpected output frame shape: {tuple(frames.shape)}")

    # Drop the first frame (reference image frame) to keep only generated motion frames.
    write_video(
        frames[1:],
        args.output_video_path,
        fps=args.fps,
        prefer_gpu_video_codec=args.prefer_gpu_video_codec,
    )
    audio_attached = False
    if args.keep_input_audio:
        audio_attached = attach_audio_from_driving_video(args.output_video_path, args.driving_video_path)
    print(f"Saved generated video: {args.output_video_path}")
    print(f"Saved DWPose preview video: {args.pose_preview_path}")
    return {
        "output_video_path": args.output_video_path,
        "pose_preview_path": args.pose_preview_path,
        "num_generated_frames": int(frames[1:].shape[0]),
        "audio_attached": audio_attached,
    }


if __name__ == "__main__":
    run_inference(parse_args())
