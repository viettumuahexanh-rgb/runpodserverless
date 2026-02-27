from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


def _create_video_reader(video_path: str):
    # Prefer GPU decode when decord has CUDA support.
    try:
        return decord.VideoReader(video_path, ctx=decord.gpu(0))
    except Exception:
        return decord.VideoReader(video_path, ctx=decord.cpu(0))


def get_video_pose(video_path: str, ref_image: np.ndarray, sample_stride: int = 1):
    ref_pose = dwprocessor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [
        i
        for i in ref_keypoint_id
        if len(ref_pose["bodies"]["subset"]) > 0 and ref_pose["bodies"]["subset"][0][i] >= 0.0
    ]
    ref_body = ref_pose["bodies"]["candidate"][ref_keypoint_id]

    height, width, _ = ref_image.shape

    vr = _create_video_reader(video_path)
    sample_stride *= max(1, int(vr.get_avg_fps() / 24))
    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    detected_poses = [dwprocessor(frm) for frm in tqdm(frames, desc="DWPose")]
    dwprocessor.release_memory()

    detected_bodies = np.stack(
        [p["bodies"]["candidate"] for p in detected_poses if p["bodies"]["candidate"].shape[0] == 18]
    )[:, ref_keypoint_id]
    ay, by = np.polyfit(
        detected_bodies[:, :, 1].flatten(),
        np.tile(ref_body[:, 1], len(detected_bodies)),
        1,
    )
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    for detected_pose in detected_poses:
        detected_pose["bodies"]["candidate"] = detected_pose["bodies"]["candidate"] * a + b
        detected_pose["faces"] = detected_pose["faces"] * a + b
        detected_pose["hands"] = detected_pose["hands"] * a + b
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
    return np.stack(output_pose)


def get_image_pose(ref_image):
    height, width, _ = ref_image.shape
    ref_pose = dwprocessor(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)
