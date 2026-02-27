from __future__ import annotations

import base64
import os
import shutil
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import requests
import runpod
from huggingface_hub import hf_hub_download

from mimicmotion_infer_6gb import run_inference


BASE_DIR = Path(__file__).resolve().parent.parent
MIMICMOTION_DIR = Path(os.getenv("MIMICMOTION_DIR", str(BASE_DIR / "MimicMotion")))


def _profile_defaults(profile: str) -> dict:
    name = (profile or "").strip().lower()
    if name in {"low", "low_vram", "6gb", "vram6"}:
        return {
            "profile": "low_vram",
            "width": 512,
            "height": 512,
            "sample_stride": 2,
            "fps": 7,
            "steps": 25,
            "guidance_scale": 2.0,
            "noise_aug_strength": 0.02,
            "tile_size": 16,
            "tile_overlap": 4,
            "vae_decode_chunk_size": 2,
            "disable_fp8": False,
        }
    if name in {"balanced", "medium"}:
        return {
            "profile": "balanced",
            "width": 576,
            "height": 1024,
            "sample_stride": 1,
            "fps": 12,
            "steps": 30,
            "guidance_scale": 2.0,
            "noise_aug_strength": 0.0,
            "tile_size": 24,
            "tile_overlap": 6,
            "vae_decode_chunk_size": 4,
            "disable_fp8": True,
        }
    return {
        "profile": "full",
        "width": 576,
        "height": 1024,
        "sample_stride": 1,
        "fps": 15,
        "steps": 35,
        "guidance_scale": 2.0,
        "noise_aug_strength": 0.0,
        "tile_size": 32,
        "tile_overlap": 8,
        "vae_decode_chunk_size": 8,
        "disable_fp8": True,
    }


def _coerce_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _download_file(url: str, dst_path: Path) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(dst_path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    out_file.write(chunk)
    return dst_path


def _decode_data_uri(data_uri: str, dst_path: Path) -> Path:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    _, encoded = data_uri.split(",", 1)
    decoded = base64.b64decode(encoded)
    with open(dst_path, "wb") as out_file:
        out_file.write(decoded)
    return dst_path


def _resolve_input_path(
    value: str | None,
    dst_path: Path,
) -> Path:
    if not value:
        raise ValueError("Input value is missing.")

    if value.startswith("http://") or value.startswith("https://"):
        return _download_file(value, dst_path)

    if value.startswith("data:"):
        return _decode_data_uri(value, dst_path)

    local_path = Path(value)
    if not local_path.exists():
        raise FileNotFoundError(f"Input file not found: {value}")
    if local_path.resolve() == dst_path.resolve():
        return local_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_path, dst_path)
    return dst_path


def _pick_input(job_input: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = job_input.get(key)
        if value:
            return str(value)
    return None


def _ensure_models_root() -> Path:
    preferred = Path(os.getenv("MODELS_ROOT", "/runpod-volume/models"))
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return preferred
    except Exception:
        fallback = Path("/tmp/models")
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _ensure_checkpoint(models_root: Path) -> Path:
    ckpt_path = models_root / "MimicMotion_1-1.pth"
    if ckpt_path.exists():
        return ckpt_path
    hf_hub_download(
        repo_id="tencent/MimicMotion",
        filename="MimicMotion_1-1.pth",
        local_dir=str(models_root),
        local_dir_use_symlinks=False,
    )
    return ckpt_path


def _ensure_dwpose(models_root: Path) -> Path:
    dwpose_dir = models_root / "DWPose"
    dwpose_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "yolox_l.onnx": "yolox_l.onnx",
        "dw-ll_ucoco_384.onnx": "dw-ll_ucoco_384.onnx",
    }
    for local_name, remote_name in required.items():
        file_path = dwpose_dir / local_name
        if file_path.exists():
            continue
        hf_hub_download(
            repo_id="yzd-v/DWPose",
            filename=remote_name,
            local_dir=str(dwpose_dir),
            local_dir_use_symlinks=False,
        )
    return dwpose_dir


def _link_dwpose_into_mimicmotion(dwpose_dir: Path) -> None:
    mm_models_dir = MIMICMOTION_DIR / "models"
    mm_models_dir.mkdir(parents=True, exist_ok=True)
    mm_dwpose = mm_models_dir / "DWPose"
    if mm_dwpose.exists():
        return
    try:
        os.symlink(dwpose_dir, mm_dwpose, target_is_directory=True)
    except Exception:
        shutil.copytree(dwpose_dir, mm_dwpose, dirs_exist_ok=True)


def _upload_if_requested(file_path: Path, put_url: str | None) -> dict | None:
    if not put_url:
        return None
    with open(file_path, "rb") as input_file:
        response = requests.put(put_url, data=input_file, timeout=600)
    response.raise_for_status()
    return {
        "put_url": put_url,
        "status_code": response.status_code,
    }


def _file_to_base64(path: Path) -> str:
    with open(path, "rb") as input_file:
        return base64.b64encode(input_file.read()).decode("utf-8")


def handler(job: dict) -> dict:
    job_input = job.get("input", {})
    job_id = str(job.get("id") or uuid.uuid4().hex)
    work_dir = Path("/tmp/runpod-jobs") / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        profile_name = str(job_input.get("quality_profile") or os.getenv("DEFAULT_QUALITY_PROFILE", "full"))
        profile_cfg = _profile_defaults(profile_name)

        models_root = _ensure_models_root()
        ckpt_path = Path(job_input.get("ckpt_path") or _ensure_checkpoint(models_root))
        dwpose_dir = Path(job_input.get("dwpose_dir") or _ensure_dwpose(models_root))
        _link_dwpose_into_mimicmotion(dwpose_dir)

        ref_value = _pick_input(job_input, ["ref_image_url", "ref_image_path", "ref_image_base64"])
        drive_value = _pick_input(job_input, ["driving_video_url", "driving_video_path", "driving_video_base64"])

        if not ref_value or not drive_value:
            raise ValueError("Provide ref image and driving video via *_url, *_path, or *_base64 keys.")

        ref_ext = Path(urlparse(ref_value).path).suffix or ".png"
        video_ext = Path(urlparse(drive_value).path).suffix or ".mp4"
        ref_image_path = _resolve_input_path(ref_value, work_dir / f"ref{ref_ext}")
        driving_video_path = _resolve_input_path(drive_value, work_dir / f"drive{video_ext}")

        output_video_path = Path(job_input.get("output_video_path") or (work_dir / "output.mp4"))
        pose_preview_path = Path(job_input.get("pose_preview_path") or (work_dir / "pose_preview.mp4"))

        args = SimpleNamespace(
            base_model_path=str(job_input.get("base_model_path") or os.getenv("BASE_MODEL_PATH", "stabilityai/stable-video-diffusion-img2vid-xt")),
            ckpt_path=str(ckpt_path),
            ref_image_path=str(ref_image_path),
            driving_video_path=str(driving_video_path),
            output_video_path=str(output_video_path),
            pose_preview_path=str(pose_preview_path),
            height=_coerce_int(job_input.get("height"), profile_cfg["height"]),
            width=_coerce_int(job_input.get("width"), profile_cfg["width"]),
            batch_size=1,
            sample_stride=_coerce_int(job_input.get("sample_stride"), profile_cfg["sample_stride"]),
            fps=_coerce_int(job_input.get("fps"), profile_cfg["fps"]),
            steps=_coerce_int(job_input.get("steps"), profile_cfg["steps"]),
            guidance_scale=_coerce_float(job_input.get("guidance_scale"), profile_cfg["guidance_scale"]),
            noise_aug_strength=_coerce_float(job_input.get("noise_aug_strength"), profile_cfg["noise_aug_strength"]),
            tile_size=_coerce_int(job_input.get("tile_size"), profile_cfg["tile_size"]),
            tile_overlap=_coerce_int(job_input.get("tile_overlap"), profile_cfg["tile_overlap"]),
            vae_decode_chunk_size=_coerce_int(job_input.get("vae_decode_chunk_size"), profile_cfg["vae_decode_chunk_size"]),
            seed=_coerce_int(job_input.get("seed"), 42),
            disable_fp8=_coerce_bool(job_input.get("disable_fp8"), profile_cfg["disable_fp8"]),
        )

        old_cwd = Path.cwd()
        os.chdir(MIMICMOTION_DIR)
        try:
            infer_result = run_inference(args)
        finally:
            os.chdir(old_cwd)

        output_upload = _upload_if_requested(output_video_path, job_input.get("output_put_url"))
        pose_upload = _upload_if_requested(pose_preview_path, job_input.get("pose_put_url"))

        return_base64 = _coerce_bool(job_input.get("return_base64"), False)

        result = {
            "status": "ok",
            "job_id": job_id,
            "output_video_path": str(output_video_path),
            "pose_preview_path": str(pose_preview_path),
            "output_video_bytes": output_video_path.stat().st_size,
            "pose_preview_bytes": pose_preview_path.stat().st_size,
            "fp8_enabled": not args.disable_fp8,
            "config": {
                "quality_profile": profile_cfg["profile"],
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "sample_stride": args.sample_stride,
                "fps": args.fps,
                "vae_decode_chunk_size": args.vae_decode_chunk_size,
            },
            "inference": infer_result,
        }
        if output_upload:
            result["output_upload"] = output_upload
        if pose_upload:
            result["pose_upload"] = pose_upload
        if return_base64:
            result["output_video_base64"] = _file_to_base64(output_video_path)
            result["pose_preview_base64"] = _file_to_base64(pose_preview_path)
        return result
    except Exception as exc:
        return {
            "status": "error",
            "job_id": job_id,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
