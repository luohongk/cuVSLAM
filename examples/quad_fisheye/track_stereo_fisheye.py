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
import json
import numpy as np
import cv2
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from scipy.spatial.transform import Rotation

# generate pseudo-random colour from integer identifier for visualization
def color_from_id(identifier): return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


def to_distortion_model(distortion: str) -> vslam.Distortion.Model:
    """Convert string distortion model name to vslam.Distortion.Model enum."""
    distortion_models = {
        'pinhole': vslam.Distortion.Model.Pinhole,
        'fisheye': vslam.Distortion.Model.Fisheye,
        'equidistant': vslam.Distortion.Model.Fisheye,
        'brown': vslam.Distortion.Model.Brown,
        'polynomial': vslam.Distortion.Model.Polynomial
    }
    if distortion not in distortion_models:
        raise ValueError(f"Unknown distortion model: {distortion}")
    return distortion_models[distortion]


def transform_to_pose(transform_16):
    """Convert a 4x4 transformation matrix to a vslam.Pose object."""
    transform = np.array(transform_16).reshape([-1, 4])
    # OpenGL to OpenCV conversion
    K = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    rotation_opencv = K @ transform[:3, :3] @ K.T
    translation_opencv = K @ transform[:3, 3]
    rotation_quat = Rotation.from_matrix(rotation_opencv).as_quat()
    return vslam.Pose(rotation=rotation_quat, translation=translation_opencv)


def create_stereo_camera_pair(focal, principal, size, baseline_x, cam_idx_left, cam_idx_right):
    """Create a stereo camera pair with specified baseline."""
    cameras = []
    
    # Left camera (reference)
    cam_left = vslam.Camera()
    cam_left.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
    cam_left.focal = focal
    cam_left.principal = principal
    cam_left.size = size
    cam_left.rig_from_camera = vslam.Pose(
        rotation=[0, 0, 0, 1],
        translation=[0, 0, 0]
    )
    cameras.append(cam_left)
    
    # Right camera (offset by baseline)
    cam_right = vslam.Camera()
    cam_right.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
    cam_right.focal = focal
    cam_right.principal = principal
    cam_right.size = size
    cam_right.rig_from_camera = vslam.Pose(
        rotation=[0, 0, 0, 1],
        translation=[baseline_x, 0, 0]
    )
    cameras.append(cam_right)
    
    return cameras


def create_fisheye_camera(focal, principal, size, translation, rotation_quat):
    """Create a fisheye camera."""
    cam = vslam.Camera()
    cam.distortion = vslam.Distortion(vslam.Distortion.Model.Fisheye, [0.0, 0.0, 0.0, 0.0])
    cam.focal = focal
    cam.principal = principal
    cam.size = size
    cam.rig_from_camera = vslam.Pose(rotation=rotation_quat, translation=translation)
    return cam


# =============================================================================
# 配置区域 - 请根据你的相机参数修改以下内容
# =============================================================================

# 相机列表名称
CAMERA_LIST = ['stereo_left', 'stereo_right', 'fisheye2', 'fisheye3']  # cam0-cam1=双目, cam2-cam3=鱼眼

# 数据路径
data_path = '/root/datasets/cuvslam_data/quad_fisheye_mydata'

# -----------------------------------------------------------------------------
# 双目相机参数 (cam0-cam1)
# -----------------------------------------------------------------------------
# 左相机参数 (cam0)
stereo_left_focal = [395.00110, 394.77479]  # fx, fy
stereo_left_principal = [536.16177, 673.65536]  # cx, cy
stereo_left_size = [512, 512]  # height, width

# 右相机参数 (cam1)
stereo_right_focal = [393.55340, 393.28716]  # fx, fy
stereo_right_principal = [542.22069, 657.45038]  # cx, cy
stereo_right_size = [512, 512]  # height, width

# 双目相机外参 (右相机相对于左相机)
# translation: 右相机光心相对于左相机的位置 (单位：米)
stereo_right_translation = [-0.03559, -0.12664, -0.06172]  # x, y, z
# rotation: 右相机相对于左相机的旋转四元数 (x, y, z, w)
# 如果双目已校正，使用单位四元数 [0, 0, 0, 1]
stereo_right_rotation = [0.00178, -0.43292, 0.86628, 0.24927]

# -----------------------------------------------------------------------------
# 鱼眼相机参数 (cam2-cam3)
# -----------------------------------------------------------------------------
# cam2 的外参 (相对 rig 原点的位置)
fisheye2_translation = [0.06754, -0.14641, -0.11441]  # x, y, z
fisheye2_rotation = [0.61095, 0.00173, 0.35044, 0.70988]  # quaternion (x, y, z, w)

# cam3 的外参
fisheye3_translation = [-0.13087, -0.07323, -0.22308]
fisheye3_rotation = [-0.00368, 0.83774, -0.25562, -0.48253]

# cam2 内参
fisheye2_focal = [395.45425, 395.55380]
fisheye2_principal = [545.85606, 657.97980]
fisheye2_size = [1088, 1280]  # height, width (actual on-disk size)
fisheye2_distortion = [-0.05836, 0.00079, -0.00629, 0.00239]  # k1,k2,k3,k4 (equidistant)

# cam3 内参
fisheye3_focal = [395.72395, 395.26038]
fisheye3_principal = [538.99676, 641.45007]
fisheye3_size = [1088, 1280]  # height, width (actual on-disk size)
fisheye3_distortion = [-0.05622, -0.00532, -0.00007, 0.00012]  # k1,k2,k3,k4 (equidistant)

# =============================================================================
# 图像预处理配置：所有相机统一输出 OUTPUT_SIZE × OUTPUT_SIZE
# =============================================================================
OUTPUT_SIZE = 512

# --- 双目预处理：中心裁剪宽度 1280→1088（变正方形），再 resize 到 512×512 ---
_S_SRC_H = 1088
_S_SRC_W = 1280
_S_CROP_X = (_S_SRC_W - _S_SRC_H) // 2   # = 96，从左侧裁掉的列数
_S_SCALE  = OUTPUT_SIZE / _S_SRC_H        # = 512/1088

def _stereo_new_intrinsics(focal, principal):
    """将双目内参映射到 crop+resize 后的坐标系。"""
    fx = focal[0] * _S_SCALE
    fy = focal[1] * _S_SCALE
    cx = (principal[0] - _S_CROP_X) * _S_SCALE
    cy = principal[1] * _S_SCALE
    return [fx, fy], [cx, cy]

stereo_left_focal_proc,  stereo_left_principal_proc  = _stereo_new_intrinsics(stereo_left_focal,  stereo_left_principal)
stereo_right_focal_proc, stereo_right_principal_proc = _stereo_new_intrinsics(stereo_right_focal, stereo_right_principal)

def preprocess_stereo(img):
    """中心裁剪 + resize → OUTPUT_SIZE×OUTPUT_SIZE (uint8 HWC)。"""
    cropped = img[:, _S_CROP_X : _S_CROP_X + _S_SRC_H]          # H×1088
    return cv2.resize(cropped, (OUTPUT_SIZE, OUTPUT_SIZE),
                      interpolation=cv2.INTER_LINEAR)

# --- 鱼眼预处理：cv2.fisheye undistort → resize 到 512×512 ---
def _build_fisheye_map(focal, principal, src_hw, distortion):
    """
    预计算 cv2.fisheye undistort 映射表，返回 (map1, map2, new_focal, new_principal)。
    src_hw : (H, W) 原始图像尺寸
    新内参：焦距按 OUTPUT_SIZE/max(H,W) 缩放，主点固定在输出图中心，确保主点坐标恒为正。
    """
    H, W = src_hw
    K = np.array([[focal[0], 0., principal[0]],
                  [0., focal[1], principal[1]],
                  [0., 0., 1.]], dtype=np.float64)
    D = np.array(distortion, dtype=np.float64).reshape(4, 1)

    # 按最大边缩放焦距，主点置于输出图中心
    scale = OUTPUT_SIZE / max(H, W)
    new_fx = focal[0] * scale
    new_fy = focal[1] * scale
    cx_new = OUTPUT_SIZE / 2.0
    cy_new = OUTPUT_SIZE / 2.0
    new_K = np.array([[new_fx, 0., cx_new],
                      [0., new_fy, cy_new],
                      [0., 0., 1.]], dtype=np.float64)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3, dtype=np.float64), new_K,
        (OUTPUT_SIZE, OUTPUT_SIZE), cv2.CV_16SC2
    )
    new_focal     = [float(new_K[0, 0]), float(new_K[1, 1])]
    new_principal = [float(new_K[0, 2]), float(new_K[1, 2])]
    return map1, map2, new_focal, new_principal

_fe2_map1, _fe2_map2, fisheye2_focal_proc, fisheye2_principal_proc = \
    _build_fisheye_map(fisheye2_focal, fisheye2_principal, fisheye2_size, fisheye2_distortion)

_fe3_map1, _fe3_map2, fisheye3_focal_proc, fisheye3_principal_proc = \
    _build_fisheye_map(fisheye3_focal, fisheye3_principal, fisheye3_size, fisheye3_distortion)

def preprocess_fisheye(img, map1, map2):
    """利用预计算映射表做鱼眼畸变矫正，输出 OUTPUT_SIZE×OUTPUT_SIZE。"""
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

# 所有相机处理后统一尺寸 [height, width]
PROC_SIZE = [OUTPUT_SIZE, OUTPUT_SIZE]

print(f"Stereo left  processed intrinsics: focal={[f'{v:.3f}' for v in stereo_left_focal_proc]}, "
      f"principal={[f'{v:.3f}' for v in stereo_left_principal_proc]}")
print(f"Stereo right processed intrinsics: focal={[f'{v:.3f}' for v in stereo_right_focal_proc]}, "
      f"principal={[f'{v:.3f}' for v in stereo_right_principal_proc]}")
print(f"Fisheye2     processed intrinsics: focal={[f'{v:.3f}' for v in fisheye2_focal_proc]}, "
      f"principal={[f'{v:.3f}' for v in fisheye2_principal_proc]}")
print(f"Fisheye3     processed intrinsics: focal={[f'{v:.3f}' for v in fisheye3_focal_proc]}, "
      f"principal={[f'{v:.3f}' for v in fisheye3_principal_proc]}")

# =============================================================================


def build_camera_rig():
    """Build camera rig with stereo pair and fisheye cameras (all using processed 512×512 images)."""
    cameras = []

    # 左相机 (cam0) - 参考相机，原点
    cam_left = vslam.Camera()
    cam_left.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
    cam_left.focal = stereo_left_focal_proc
    cam_left.principal = stereo_left_principal_proc
    cam_left.size = PROC_SIZE
    cam_left.rig_from_camera = vslam.Pose(rotation=[0, 0, 0, 1], translation=[0, 0, 0])
    cameras.append(cam_left)

    # 右相机 (cam1)
    cam_right = vslam.Camera()
    cam_right.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
    cam_right.focal = stereo_right_focal_proc
    cam_right.principal = stereo_right_principal_proc
    cam_right.size = PROC_SIZE
    cam_right.rig_from_camera = vslam.Pose(rotation=stereo_right_rotation, translation=stereo_right_translation)
    cameras.append(cam_right)

    # 鱼眼相机 cam2（undistort 后为 Pinhole）
    cam_fisheye2 = vslam.Camera()
    cam_fisheye2.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
    cam_fisheye2.focal = fisheye2_focal_proc
    cam_fisheye2.principal = fisheye2_principal_proc
    cam_fisheye2.size = PROC_SIZE
    cam_fisheye2.rig_from_camera = vslam.Pose(rotation=fisheye2_rotation, translation=fisheye2_translation)
    cameras.append(cam_fisheye2)

    # 鱼眼相机 cam3（undistort 后为 Pinhole）
    cam_fisheye3 = vslam.Camera()
    cam_fisheye3.distortion = vslam.Distortion(vslam.Distortion.Model.Pinhole, [])
    cam_fisheye3.focal = fisheye3_focal_proc
    cam_fisheye3.principal = fisheye3_principal_proc
    cam_fisheye3.size = PROC_SIZE
    cam_fisheye3.rig_from_camera = vslam.Pose(rotation=fisheye3_rotation, translation=fisheye3_translation)
    cameras.append(cam_fisheye3)

    return cameras


# Get number of frames
if os.path.exists(os.path.join(data_path, f'image_{CAMERA_LIST[0]}')):
    num_frames = len(os.listdir(os.path.join(data_path, f'image_{CAMERA_LIST[0]}')))
else:
    print(f"Error: Data path not found: {data_path}")
    print("Please modify 'data_path' in the script to point to your dataset directory.")
    exit(1)


### setup rerun visualizer
rr.init('stereo_fisheye', strict=True, spawn=True)  # launch re-run instance
# setup rerun views - 4 cameras in horizontal layout
rr.send_blueprint(rrb.Blueprint(rrb.TimePanel(state="collapsed"),
                                rrb.Vertical(
                                    contents=[
                                        rrb.Horizontal(
                                            contents=[rrb.Spatial2DView(origin=f'car/cam{i}', name=f'cam{i}') for i in range(4)]),
                                        rrb.Spatial3DView(name="3D", defaults=[rr.components.ImagePlaneDistance(0.5)])
                                        ]
                                    ),
                                ))
# setup coordinate basis for root, cuvslam uses right-hand system with  X-right, Y-down, Z-forward
rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# Build camera rig
cameras = build_camera_rig()

# Debug: print camera extrinsics
print("\n=== Camera Configuration (processed 512×512) ===")
print("Stereo Left (cam0):")
print(f"  focal={stereo_left_focal_proc}, principal={stereo_left_principal_proc}, size={PROC_SIZE}")
print(f"  Translation: [0.0000, 0.0000, 0.0000]")

print("Stereo Right (cam1):")
print(f"  focal={stereo_right_focal_proc}, principal={stereo_right_principal_proc}, size={PROC_SIZE}")
print(f"  Translation: [{stereo_right_translation[0]:.4f}, {stereo_right_translation[1]:.4f}, {stereo_right_translation[2]:.4f}]")
print(f"  Rotation: {stereo_right_rotation}")

print("Fisheye (cam2):")
print(f"  focal={fisheye2_focal_proc}, principal={fisheye2_principal_proc}, size={PROC_SIZE}")
print(f"  Translation: [{fisheye2_translation[0]:.4f}, {fisheye2_translation[1]:.4f}, {fisheye2_translation[2]:.4f}]")

print("Fisheye (cam3):")
print(f"  focal={fisheye3_focal_proc}, principal={fisheye3_principal_proc}, size={PROC_SIZE}")
print(f"  Translation: [{fisheye3_translation[0]:.4f}, {fisheye3_translation[1]:.4f}, {fisheye3_translation[2]:.4f}]")

# Set up VSLAM rig and tracker
rig = vslam.Rig()
rig.cameras = cameras

# Configure tracker
# rectified_stereo_camera=True: 双目已校正，使用快速跟踪
# rectified_stereo_camera=False: 未校正，需要完整畸变处理
cfg = vslam.Tracker.OdometryConfig(
    multicam_mode=vslam.Tracker.MulticameraMode.Performance,
    odometry_mode=vslam.Tracker.OdometryMode.Multicamera,
    rectified_stereo_camera=False,  # 双目未校正（使用真实外参）
    use_gpu=True,
    async_sba=True
)

tracker = vslam.Tracker(rig, cfg)

trajectory = []

print(f"\nTotal frames to process: {num_frames}")

# Process each frame
for frame_id in range(num_frames):
    timestamp = frame_id
    try:
        raw = [np.asarray(Image.open(os.path.join(data_path, f'image_{cam}', f'{frame_id:06d}.png')))
               for cam in CAMERA_LIST]
    except FileNotFoundError as e:
        print(f"Error: Missing image file for frame {frame_id}: {e}")
        continue

    # 图像预处理：统一输出 OUTPUT_SIZE×OUTPUT_SIZE
    images = [
        preprocess_stereo(raw[0]),                      # cam0 双目左
        preprocess_stereo(raw[1]),                      # cam1 双目右
        preprocess_fisheye(raw[2], _fe2_map1, _fe2_map2),  # cam2 鱼眼矫正
        preprocess_fisheye(raw[3], _fe3_map1, _fe3_map2),  # cam3 鱼眼矫正
    ]
    
    # do multicamera visual tracking
    odom_pose_estimate, _ = tracker.track(timestamp, images)

    if odom_pose_estimate.world_from_rig is None:
        print(f"Warning: Failed to track frame {frame_id}")
        continue

    # Get current pose
    odom_pose = odom_pose_estimate.world_from_rig.pose

    # get visualization data
    observations = [tracker.get_last_observations(i) for i in range(len(CAMERA_LIST))]
    landmarks = tracker.get_last_landmarks()
    final_landmarks = tracker.get_final_landmarks()
    # prepare visualization data
    observations_uv = [[[o.u, o.v] for o in obs_instance] for obs_instance in observations]
    observations_colors = [[color_from_id(o.id) for o in obs_instance] for obs_instance in observations]
    landmark_xyz = [l.coords for l in landmarks]
    landmarks_colors = [color_from_id(l.id) for l in landmarks]
    trajectory.append(odom_pose.translation)
    # send results to rerun for visualization
    rr.set_time_sequence('frame', frame_id)
    rr.log('trajectory', rr.LineStrips3D(trajectory))
    rr.log('final_landmarks', rr.Points3D(list(final_landmarks.values()), radii=0.01))
    rr.log('car', rr.Transform3D(translation=odom_pose.translation, quaternion=odom_pose.rotation))
    rr.log('car/body', rr.Boxes3D(centers=[0, 0.3 / 2, 0], sizes=[[0.35, 0.3, 0.66]]))
    rr.log('car/landmarks_center', rr.Points3D(landmark_xyz, radii=0.02, colors=landmarks_colors))

    # Debug: print observations per camera
    if frame_id % 10 == 0:
        print(f"Frame {frame_id}: pos=[{odom_pose.translation[0]:.3f}, {odom_pose.translation[1]:.3f}, {odom_pose.translation[2]:.3f}], "
              f"landmarks={len(landmarks)}, obs per cam={[len(obs) for obs in observations]}")

    for i in range(len(cameras)):
        rr.log(f'car/cam{i}/image', rr.Image(images[i]).compress(jpeg_quality=80))
        rr.log(f'car/cam{i}/observations', rr.Points2D(observations_uv[i], radii=5, colors=observations_colors[i]))
        
        # Log camera poses
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
