import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..modules.unet import UNetSpatioTemporalConditionModel
from ..modules.pose_net import PoseNet
from ..pipelines.pipeline_mimicmotion import MimicMotionPipeline

logger = logging.getLogger(__name__)


class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet")
        )
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=torch.float16, variant="fp16"
        )
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder", torch_dtype=torch.float16, variant="fp16"
        )
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(base_model_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(base_model_path, subfolder="feature_extractor")
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0])


def create_pipeline(infer_config, device):
    mimicmotion_models = MimicMotionModel(infer_config.base_model_path)
    try:
        checkpoint = torch.load(infer_config.ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(infer_config.ckpt_path, map_location="cpu")
    mimicmotion_models.load_state_dict(checkpoint, strict=False)
    for key in checkpoint.keys():
        if not any(key.startswith(prefix) for prefix in ["unet", "vae", "image_encoder", "pose_net"]):
            logger.warning(f"Unexpected key in checkpoint: {key}")
    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae,
        image_encoder=mimicmotion_models.image_encoder,
        unet=mimicmotion_models.unet,
        scheduler=mimicmotion_models.noise_scheduler,
        feature_extractor=mimicmotion_models.feature_extractor,
        pose_net=mimicmotion_models.pose_net,
    )
    return pipeline
