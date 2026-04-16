# ==============================================================================
# cuVSLAM — 头套四鱼眼相机追踪脚本
# 相机布局：环绕头套约 90° 间隔，共 4 路鱼眼相机
# 坐标系：OpenCV / ROS 约定（相机朝 +Z，Y 朝下，右手系）
# 图像存储：image_cam0~image_cam3 / 000000.png
# ==============================================================================

import cuvslam as vslam
import os
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image
from scipy.spatial.transform import Rotation

# ==============================================================================
# 工具函数
# ==============================================================================

def color_from_id(identifier):
    """从整数 ID 生成伪随机颜色，用于可视化。"""
    return [(identifier * 17) % 256, (identifier * 31) % 256, (identifier * 47) % 256]


def opengl_to_opencv(R_mat: np.ndarray, t_vec: np.ndarray):
    """
    将 OpenGL 坐标系的外参转换到 OpenCV/ROS 坐标系。

    OpenGL 约定：X右，Y上，Z朝向观察者（相机朝 -Z）
    OpenCV 约定：X右，Y下，Z向前（相机朝 +Z）
    转换矩阵 K = diag(1, -1, -1)：
        R_cv = K @ R_gl @ K^T
        t_cv = K @ t_gl
    """
    K = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
    return K @ R_mat @ K.T, K @ np.array(t_vec, dtype=float)


def opencv_pose_to_vslam(R_mat: np.ndarray, t_vec: np.ndarray) -> vslam.Pose:
    """
    将 OpenCV/ROS 坐标系下的旋转矩阵 + 平移向量转换为 vslam.Pose。

    参数说明：
      R_mat : (3,3) 旋转矩阵，表示 cam_i 相对于 cam0 的旋转
              即  p_cam0 = R_mat @ p_cami + t_vec
      t_vec : (3,) 平移向量（单位：米）

    cuVSLAM 的 rig_from_camera 含义是"从相机坐标系到 rig 坐标系的变换"，
    即 p_rig = R @ p_cam + t，与 OpenCV 外参方向相同，无需额外取逆。
    """
    quat_xyzw = Rotation.from_matrix(R_mat).as_quat()   # [x, y, z, w]
    return vslam.Pose(rotation=quat_xyzw.tolist(), translation=t_vec.tolist())


# ==============================================================================
# ★ 请在此处填入你的相机标定参数 ★
# ------------------------------------------------------------------------------
# 每个字段说明：
#   focal     : [fx, fy]         单位：像素
#   principal : [cx, cy]         单位：像素
#   size      : [height, width]  单位：像素（注意 height 在前！）
#   dist      : [k1, k2, k3, k4] Kannala-Brandt (等距) 鱼眼模型
#   R         : (3,3) 旋转矩阵   相对于 cam0（参考相机）
#   t         : [x, y, z]        平移（米），相对于 cam0
#
# ★ 坐标系说明（重要！）：
#   EXTRINSIC_COORD = 'opengl'  → R/t 来自 OpenGL 约定（X右,Y上,Z朝观察者）
#                                  例如 NVIDIA EDEX 格式、Blender 导出
#   EXTRINSIC_COORD = 'opencv'  → R/t 来自 OpenCV/ROS 约定（X右,Y下,Z向前）
#                                  例如 Kalibr、OpenCV stereoCalibrate 输出
#
# ⚠️  cam0 是参考相机，其 R = 单位阵，t = [0,0,0]。
# ==============================================================================

# ★ 根据你的标定工具修改这里：'opengl' 或 'opencv'
EXTRINSIC_COORD = 'opengl'

CAMERAS_CFG = {
    # ------------------------------------------------------------------
    # cam0 — 参考相机（前方，或任意选一个作为参考，外参固定为 identity）
    # ------------------------------------------------------------------
    'cam0': {
        'focal'    : [395.00110, 394.77479],
        'principal': [536.16177, 673.65536],
        'size'     : [1088, 1280],   # [height, width]
        'dist'     : [-0.05762, 0.00294, -0.00663, 0.00178],
        'R'        : np.eye(3),
        't'        : [0.0, 0.0, 0.0],
    },
    # ------------------------------------------------------------------
    # cam1
    # ------------------------------------------------------------------
    'cam1': {
        'focal'    : [395.45425, 395.55380],
        'principal': [545.85606, 657.97980],
        'size'     : [1088, 1280],   # [height, width]
        'dist'     : [-0.05836, 0.00079, -0.00629, 0.00239],
        'R'        : np.array([
            [ 0.75438,  0.49965,  0.42575],
            [-0.49543,  0.00786,  0.86861],
            [ 0.43066, -0.86619,  0.25347],
        ]),
        't'        : [0.07091, 0.13399, -0.12691],
    },
    # ------------------------------------------------------------------
    # cam2
    # ------------------------------------------------------------------
    'cam2': {
        'focal'    : [393.55340, 393.28716],
        'principal': [542.22069, 657.45038],
        'size'     : [1088, 1280],   # [height, width]
        'dist'     : [-0.05448, -0.00459, -0.00141, 0.00070],
        'R'        : np.array([
            [-0.87572,  0.43033,  0.21891],
            [-0.43342, -0.50089, -0.74917],
            [-0.21274, -0.75095,  0.62515],
        ]),
        't'        : [0.03684, -0.12510, -0.06409],
    },
    # ------------------------------------------------------------------
    # cam3
    # ------------------------------------------------------------------
    'cam3': {
        'focal'    : [395.72395, 395.26038],
        'principal': [538.99676, 641.45007],
        'size'     : [1088, 1280],   # [height, width]
        'dist'     : [-0.05622, -0.00532, -0.00007, 0.00012],
        'R'        : np.array([
            [-0.53430,  0.24052,  0.81035],
            [-0.25285,  0.86929, -0.42473],
            [-0.80659, -0.43184, -0.40364],
        ]),
        't'        : [0.12846, -0.06418, -0.22723],
    },
}

# ==============================================================================
# 数据集路径配置
# ==============================================================================

DATA_PATH = '/root/datasets/cuvslam_data/quad_fisheye_data'   # ← 替换为你的数据集根目录

# 验证目录存在
for cam_name in CAMERAS_CFG:
    cam_dir = os.path.join(DATA_PATH, f'image_{cam_name}')
    if not os.path.exists(cam_dir):
        raise FileNotFoundError(
            f"找不到相机目录: {cam_dir}\n"
            f"请确认 DATA_PATH 正确，且目录命名为 image_cam0 ~ image_cam3"
        )

num_frames = len(os.listdir(os.path.join(DATA_PATH, 'image_cam0')))
print(f"数据集路径: {DATA_PATH}")
print(f"检测到帧数: {num_frames}")

# ==============================================================================
# 构建 cuVSLAM 相机 Rig
# ==============================================================================

def build_camera_rig() -> list:
    cameras = []
    for cam_name, cfg in CAMERAS_CFG.items():
        cam = vslam.Camera()

        # 鱼眼畸变模型（Kannala-Brandt，等距投影，4 参数）
        cam.distortion = vslam.Distortion(
            vslam.Distortion.Model.Fisheye,
            cfg['dist']
        )
        cam.focal     = cfg['focal']
        cam.principal = cfg['principal']
        cam.size      = cfg['size']   # [height, width]

        # 外参：按坐标系约定转换后送入 cuVSLAM（需要 OpenCV 约定）
        R = np.array(cfg['R'], dtype=float)
        t = np.array(cfg['t'], dtype=float)
        if EXTRINSIC_COORD == 'opengl':
            R, t = opengl_to_opencv(R, t)
        # elif EXTRINSIC_COORD == 'opencv': 直接使用，无需转换

        cam.rig_from_camera = opencv_pose_to_vslam(R, t)
        cameras.append(cam)

        print(f"  {cam_name}: focal={cfg['focal']}, size={cfg['size']}, "
              f"t_cv={[f'{v:.4f}' for v in t]}")

    return cameras


print("\n=== 相机 Rig 配置 ===")
cameras = build_camera_rig()

rig = vslam.Rig()
rig.cameras = cameras

# ==============================================================================
# Tracker 配置
# ==============================================================================

cfg = vslam.Tracker.OdometryConfig(
    # 多相机模式：每个相机独立贡献特征点，不假设双目约束
    multicam_mode=vslam.Tracker.MulticameraMode.Performance,
    odometry_mode=vslam.Tracker.OdometryMode.Multicamera,
    rectified_stereo_camera=False,   # 非标准双目，不要开校正模式
    enable_final_landmarks_export=True,
    use_gpu=True,
    async_sba=True,
)

tracker = vslam.Tracker(rig, cfg)

# ==============================================================================
# Rerun 可视化初始化
# ==============================================================================

rr.init('helmet_quad_fisheye', strict=True, spawn=True)
rr.send_blueprint(rrb.Blueprint(
    rrb.TimePanel(state="collapsed"),
    rrb.Vertical(contents=[
        rrb.Horizontal(contents=[
            rrb.Spatial2DView(origin='helmet/cam0', name='cam0 (前)'),
            rrb.Spatial2DView(origin='helmet/cam1', name='cam1 (~90°)'),
            rrb.Spatial2DView(origin='helmet/cam2', name='cam2 (后)'),
            rrb.Spatial2DView(origin='helmet/cam3', name='cam3 (~270°)'),
        ]),
        rrb.Spatial3DView(name='3D 轨迹', defaults=[rr.components.ImagePlaneDistance(0.5)]),
    ]),
))

# cuVSLAM 使用右手系：X-right, Y-down, Z-forward
rr.log('/', rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# ==============================================================================
# 主追踪循环
# ==============================================================================

trajectory = []
cam_names  = list(CAMERAS_CFG.keys())

print(f"\n=== 开始追踪，共 {num_frames} 帧 ===\n")

for frame_id in range(num_frames):
    timestamp = frame_id   # 若有真实时间戳可替换为实际值（单位：纳秒）

    # ---------- 读取当前帧图像 ----------
    try:
        images = [
            np.asarray(Image.open(
                os.path.join(DATA_PATH, f'image_{cam}', f'{frame_id:06d}.png')
            ))          # 保持原始格式（RGB彩色或灰度均可，cuVSLAM自动处理）
            for cam in cam_names
        ]
    except FileNotFoundError as e:
        print(f"[警告] 帧 {frame_id} 缺少图像文件，跳过：{e}")
        continue

    # ---------- 追踪 ----------
    odom_pose_estimate, _ = tracker.track(timestamp, images)

    if odom_pose_estimate.world_from_rig is None:
        print(f"[警告] 帧 {frame_id} 追踪失败（特征点不足？），跳过")
        continue

    odom_pose = odom_pose_estimate.world_from_rig.pose

    # ---------- 获取可视化数据 ----------
    observations    = [tracker.get_last_observations(i) for i in range(len(cameras))]
    landmarks       = tracker.get_last_landmarks()
    final_landmarks = tracker.get_final_landmarks()

    obs_uv     = [[[o.u, o.v] for o in obs] for obs in observations]
    obs_colors = [[color_from_id(o.id) for o in obs] for obs in observations]
    lm_xyz     = [l.coords for l in landmarks]
    lm_colors  = [color_from_id(l.id) for l in landmarks]

    trajectory.append(odom_pose.translation)

    # ---------- 打印进度 ----------
    if frame_id % 20 == 0:
        t = odom_pose.translation
        print(f"帧 {frame_id:5d} | pos=[{t[0]:7.3f}, {t[1]:7.3f}, {t[2]:7.3f}] "
              f"| 路标={len(landmarks):4d} "
              f"| 各相机观测={[len(o) for o in observations]}")

    # ---------- Rerun 可视化 ----------
    rr.set_time_sequence('frame', frame_id)

    rr.log('trajectory',           rr.LineStrips3D(trajectory))
    rr.log('final_landmarks',      rr.Points3D(list(final_landmarks.values()), radii=0.01))
    rr.log('helmet',               rr.Transform3D(
                                       translation=odom_pose.translation,
                                       quaternion=odom_pose.rotation))
    rr.log('helmet/body',          rr.Boxes3D(centers=[0, 0, 0], sizes=[[0.22, 0.25, 0.30]]))
    rr.log('helmet/landmarks',     rr.Points3D(lm_xyz, radii=0.02, colors=lm_colors))

    for i, cam_name in enumerate(cam_names):
        rr.log(f'helmet/cam{i}/image',
               rr.Image(images[i]).compress(jpeg_quality=80))
        rr.log(f'helmet/cam{i}/observations',
               rr.Points2D(obs_uv[i], radii=5, colors=obs_colors[i]))
        rr.log(f'helmet/cam{i}',
               rr.Transform3D(
                   translation=cameras[i].rig_from_camera.translation,
                   rotation=rr.Quaternion(xyzw=cameras[i].rig_from_camera.rotation),
                   from_parent=False))
        rr.log(f'helmet/cam{i}',
               rr.Pinhole(
                   image_plane_distance=1.0,
                   image_from_camera=np.array([
                       [cameras[i].focal[0], 0,                    cameras[i].principal[0]],
                       [0,                   cameras[i].focal[1],  cameras[i].principal[1]],
                       [0,                   0,                    1                      ],
                   ]),
                   width=cameras[i].size[1], height=cameras[i].size[0],
               ))

print("\n追踪完成！按 Ctrl+C 退出（Rerun 窗口保持打开）。")
import time
while True:
    time.sleep(1)
