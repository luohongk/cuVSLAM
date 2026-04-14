# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA software released under the NVIDIA Community License is intended to be used to enable
# the further development of AI and robotics technologies. Such software has been designed, tested,
# and optimized for use with NVIDIA hardware, and this License grants permission to use the software
# solely with such hardware.
# Subject to the terms of this License, NVIDIA confirms that you are free to commercially use,
# modify, and distribute the software with NVIDIA hardware. NVIDIA does not claim ownership of any
# outputs generated using the software or derivative works thereof. Any code contributions that you
# share with NVIDIA are licensed to NVIDIA as feedback under this License and may be incorporated
# in future releases without notice or attribution.
# By using, reproducing, modifying, distributing, performing, or displaying any portion or element
# of the software or derivative works thereof, you agree to be bound by this License.

import cuvslam as vslam
import os
import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

# generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier): return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


# =============================================================================
# 配置区域 - 请根据你的相机参数修改以下内容
# =============================================================================

# 相机列表名称（对应数据目录下 image_<name> 文件夹）
CAMERA_LIST = ['stereo_left', 'stereo_right']

# 数据路径
data_path = '/root/datasets/cuvslam_data/quad_fisheye_mydata'

# -----------------------------------------------------------------------------
# 双目相机参数
# -----------------------------------------------------------------------------
# 左相机内参 (cam0)
stereo_left_focal     = [395.00110, 394.77479]   # fx, fy
stereo_left_principal = [536.16177, 673.65536]   # cx, cy
stereo_left_size      = [1088, 1280]             # height, width（原始图像尺寸）

# 右相机内参 (cam1)
stereo_right_focal     = [393.55340, 393.28716]
stereo_right_principal = [542.22069, 657.45038]
stereo_right_size      = [1088, 1280]

# 右相机相对于左相机的外参
# translation: 右相机光心相对于左相机光心的位置（单位：米）
stereo_right_translation = [-0.03559, -0.12664, -0.06172]   # x, y, z
# rotation: 四元数 (x, y, z, w)；若已做双目校正则填 [0, 0, 0, 1]
stereo_right_rotation    = [0.00178, -0.43292, 0.86628, 0.24927]

# =============================================================================
# 图像预处理：中心裁剪 1280→1088 后 resize 到 OUTPUT_SIZE×OUTPUT_SIZE
# =============================================================================
OUTPUT_SIZE = 512

_SRC_H  = stereo_left_size[0]          # 1088
_SRC_W  = stereo_left_size[1]          # 1280
_CROP_X = (_SRC_W - _SRC_H) // 2      # 96，左右各裁掉的列数
_SCALE  = OUTPUT_SIZE / _SRC_H         # 512 / 1088


def _adjust_intrinsics(focal, principal):
    """将内参映射到 crop+resize 后的坐标系。"""
    fx = focal[0] * _SCALE
    fy = focal[1] * _SCALE
    cx = (principal[0] - _CROP_X) * _SCALE
    cy = principal[1] * _SCALE
    return [fx, fy], [cx, cy]


stereo_left_focal_proc,  stereo_left_principal_proc  = _adjust_intrinsics(stereo_left_focal,  stereo_left_principal)
stereo_right_focal_proc, stereo_right_principal_proc = _adjust_intrinsics(stereo_right_focal, stereo_right_principal)
PROC_SIZE = [OUTPUT_SIZE, OUTPUT_SIZE]  # height, width


def preprocess_stereo(img):
    """中心裁剪 + resize → OUTPUT_SIZE×OUTPUT_SIZE (uint8 HWC)。"""
    cropped = img[:, _CROP_X : _CROP_X + _SRC_H]
    return cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_LINEAR)


print(f"Stereo left  processed intrinsics: focal={[f'{v:.3f}' for v in stereo_left_focal_proc]}, "
      f"principal={[f'{v:.3f}' for v in stereo_left_principal_proc]}")
print(f"Stereo right processed intrinsics: focal={[f'{v:.3f}' for v in stereo_right_focal_proc]}, "
      f"principal={[f'{v:.3f}' for v in stereo_right_principal_proc]}")

# =============================================================================


def build_camera_rig():
    """Build stereo camera rig (processed 512×512 images)."""
    cameras = []

    # 左相机 (cam0) - 参考相机，原点
    cam_left = vslam.Camera()
    cam_left.distortion      = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
    cam_left.focal           = stereo_left_focal_proc
    cam_left.principal       = stereo_left_principal_proc
    cam_left.size            = PROC_SIZE
    cam_left.rig_from_camera = vslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
    cameras.append(cam_left)

    # 右相机 (cam1)
    cam_right = vslam.Camera()
    cam_right.distortion      = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
    cam_right.focal           = stereo_right_focal_proc
    cam_right.principal       = stereo_right_principal_proc
    cam_right.size            = PROC_SIZE
    cam_right.rig_from_camera = vslam.Pose(rotation=stereo_right_rotation, translation=stereo_right_translation)
    cameras.append(cam_right)

    return cameras


# =============================================================================
# 数据路径检查
# =============================================================================
if os.path.exists(os.path.join(data_path, f'image_{CAMERA_LIST[0]}')):
    num_frames = len(os.listdir(os.path.join(data_path, f'image_{CAMERA_LIST[0]}')))
else:
    print(f"Error: Data path not found: {data_path}")
    print("Please modify 'data_path' in the script to point to your dataset directory.")
    exit(1)


# =============================================================================
# Rerun 可视化初始化
# =============================================================================
rr.init('stereo_fisheye', strict=True, spawn=True)
rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Vertical(contents=[
        rrb.Horizontal(contents=[
            rrb.Spatial2DView(origin='car/cam0', name='cam0 (left)'),
            rrb.Spatial2DView(origin='car/cam1', name='cam1 (right)'),
        ]),
        rrb.Spatial3DView(name="3D", defaults=[rr.components.ImagePlaneDistance(0.5)])
    ]),
))
# cuVSLAM 使用右手系：X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# =============================================================================
# 构建相机 Rig 与 Tracker
# =============================================================================
cameras = build_camera_rig()

print("\n=== Camera Configuration (processed 512×512) ===")
print("Stereo Left (cam0):")
print(f"  focal={stereo_left_focal_proc}, principal={stereo_left_principal_proc}, size={PROC_SIZE}")
print(f"  Translation: [0.0000, 0.0000, 0.0000]")
print("Stereo Right (cam1):")
print(f"  focal={stereo_right_focal_proc}, principal={stereo_right_principal_proc}, size={PROC_SIZE}")
print(f"  Translation: [{stereo_right_translation[0]:.4f}, {stereo_right_translation[1]:.4f}, {stereo_right_translation[2]:.4f}]")
print(f"  Rotation: {stereo_right_rotation}")

rig = vslam.Rig()
rig.cameras = cameras

# rectified_stereo_camera=True: 双目已校正；=False: 使用真实外参（未校正）
cfg = vslam.Tracker.OdometryConfig(
    multicam_mode=vslam.Tracker.MulticameraMode.Performance,
    odometry_mode=vslam.Tracker.OdometryMode.Multicamera,
    rectified_stereo_camera=False,
    use_gpu=True,
    async_sba=True
)

tracker = vslam.Tracker(rig, cfg)
trajectory = []

print(f"\nTotal frames to process: {num_frames}")

# =============================================================================
# 主循环
# =============================================================================
for frame_id in range(num_frames):
    timestamp = frame_id
    try:
        raw = [np.asarray(Image.open(os.path.join(data_path, f'image_{cam}', f'{frame_id:06d}.png')))
               for cam in CAMERA_LIST]
    except FileNotFoundError as e:
        print(f"Error: Missing image file for frame {frame_id}: {e}")
        continue

    # 图像预处理：裁剪 + resize
    images = [
        preprocess_stereo(raw[0]),   # cam0 左
        preprocess_stereo(raw[1]),   # cam1 右
    ]

    odom_pose_estimate, _ = tracker.track(timestamp, images)

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame_id}")
        continue

    odom_pose = odom_pose_estimate.world_from_rig.pose

    observations     = [tracker.get_last_observations(i) for i in range(len(CAMERA_LIST))]
    landmarks        = tracker.get_last_landmarks()
    final_landmarks  = tracker.get_final_landmarks()

    observations_uv     = [[[o.u, o.v] for o in obs] for obs in observations]
    observations_colors = [[color_from_id(o.id) for o in obs] for obs in observations]
    landmark_xyz        = [l.coords for l in landmarks]
    landmarks_colors    = [color_from_id(l.id) for l in landmarks]
    trajectory.append(odom_pose.translation)

    rr.set_time_sequence('frame', frame_id)
    rr.log('trajectory',            rr.LineStrips3D(trajectory))
    rr.log('final_landmarks',       rr.Points3D(list(final_landmarks.values()), radii=0.01))
    rr.log('car',                   rr.Transform3D(translation=odom_pose.translation, quaternion=odom_pose.rotation))
    rr.log('car/body',              rr.Boxes3D(centers=[0, 0.3 / 2, 0], sizes=[[0.35, 0.3, 0.66]]))
    rr.log('car/landmarks_center',  rr.Points3D(landmark_xyz, radii=0.02, colors=landmarks_colors))

    if frame_id % 10 == 0:
        print(f"Frame {frame_id}: pos=[{odom_pose.translation[0]:.3f}, {odom_pose.translation[1]:.3f}, {odom_pose.translation[2]:.3f}], "
              f"landmarks={len(landmarks)}, obs per cam={[len(obs) for obs in observations]}")

    for i in range(len(cameras)):
        rr.log(f'car/cam{i}/image',        rr.Image(images[i]).compress(jpeg_quality=80))
        rr.log(f'car/cam{i}/observations', rr.Points2D(observations_uv[i], radii=5, colors=observations_colors[i]))
        rr.log(f'car/cam{i}', rr.Transform3D(
            translation=cameras[i].rig_from_camera.translation,
            rotation=rr.Quaternion(xyzw=cameras[i].rig_from_camera.rotation),
            from_parent=False
        ))
        rr.log(f'car/cam{i}', rr.Pinhole(
            image_plane_distance=1.,
            image_from_camera=np.array([
                [cameras[i].focal[0], 0, cameras[i].principal[0]],
                [0, cameras[i].focal[1], cameras[i].principal[1]],
                [0, 0, 1]
            ]),
            width=cameras[i].size[1],
            height=cameras[i].size[0]
        ))

print("Processing complete. Press Ctrl+C to exit.")
import time
while True:
    time.sleep(1)
