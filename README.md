# RunPod Serverless Setup (MimicMotion)

## Docker location
- Docker image definition is at `runpod_serverless/Dockerfile`.
- Worker entrypoint is `runpod_serverless/handler.py`.

## What this worker does
- Runs MimicMotion inference on GPU.
- Uses `full` quality by default on cloud GPUs (`576x1024`, higher steps).
- Preprocesses DWPose from driving video before generation.
- Attempts FP8 quantization for UNet (`optimum-quanto`), unless `disable_fp8=true`.

## Build and push
1. Build:
```bash
docker build -f runpod_serverless/Dockerfile -t <dockerhub-user>/mimicmotion-runpod:0.1.0 .
```
2. Push:
```bash
docker push <dockerhub-user>/mimicmotion-runpod:0.1.0
```

## RunPod GitHub build settings (important)
- `Dockerfile Path`: `runpod_serverless/Dockerfile`
- `Build Context`: repo root (`.`) is recommended.
- If your context is accidentally `runpod_serverless`, this Dockerfile now still works.

## Create RunPod Serverless endpoint
1. RunPod Console -> `Serverless` -> `New Endpoint`.
2. Container image: `<dockerhub-user>/mimicmotion-runpod:0.1.0`.
3. GPU: choose 24GB+ VRAM for stable runtime (16GB minimum baseline from MimicMotion docs).
4. Add network volume and mount at `/runpod-volume` (recommended).
5. Optional env vars:
- `HF_TOKEN`: token for private/gated Hugging Face repos.
- `BASE_MODEL_PATH`: default is `stabilityai/stable-video-diffusion-img2vid-xt`.
- `MODELS_ROOT`: default is `/runpod-volume/models`.
- `DEFAULT_QUALITY_PROFILE`: `full` (default) / `balanced` / `low_vram`.

## API input example
```json
{
  "input": {
    "ref_image_url": "https://.../ref.png",
    "driving_video_url": "https://.../drive.mp4",
    "quality_profile": "full",
    "steps": 35,
    "sample_stride": 1,
    "fps": 15,
    "seed": 42,
    "width": 576,
    "height": 1024,
    "vae_decode_chunk_size": 8,
    "disable_fp8": true
  }
}
```

## Input keys supported
- `ref_image_url` or `ref_image_path` or `ref_image_base64`
- `driving_video_url` or `driving_video_path` or `driving_video_base64`
- Optional tuning:
`steps`, `sample_stride`, `fps`, `seed`, `width`, `height`, `guidance_scale`, `noise_aug_strength`, `tile_size`, `tile_overlap`, `vae_decode_chunk_size`, `disable_fp8`
- Optional upload:
`output_put_url`, `pose_put_url` (pre-signed PUT URLs)
- Optional inline return:
`return_base64` (large payload risk)

## Response
- `status`: `ok` or `error`
- `output_video_path`, `pose_preview_path`
- byte sizes and used config
- optional `output_upload` / `pose_upload` info when PUT URLs are used

## Notes
- First cold start downloads large model files (several GB).
- Persisting `/runpod-volume/models` and `/runpod-volume/hf` avoids repeated downloads.
- If you use a gated base model, configure `HF_TOKEN` with accepted access.
