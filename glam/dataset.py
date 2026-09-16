import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Union
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter1d
from PIL import Image
import torchvision.transforms as T

# ==========================================
# Math Utils
# ==========================================
def ensure_quat_continuity(quaternions):
    quaternions = quaternions.copy()
    for i in range(len(quaternions) - 1):
        if np.dot(quaternions[i], quaternions[i+1]) < 0:
            quaternions[i+1] *= -1
    return quaternions

def quat_to_6d_numpy(quaternions):
    norm = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / (norm + 1e-9)
    w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    col0 = np.stack([1 - 2*(y**2 + z**2), 2*(x*y + z*w), 2*(x*z - y*w)], axis=-1)
    col1 = np.stack([2*(x*y - z*w), 1 - 2*(x**2 + z**2), 2*(y*z + x*w)], axis=-1)
    return np.concatenate([col0, col1], axis=-1)

def get_adaptive_indices(obj_traj, approach_stride, move_thresh=0.015, padding=5):
    T = len(obj_traj)
    pos = obj_traj[:, :3]
    pos_smooth = gaussian_filter1d(pos, sigma=2.0, axis=0)
    diff = np.linalg.norm(pos_smooth - pos_smooth[0], axis=1)
    moving_mask = diff > move_thresh
    if not np.any(moving_mask): grasp_idx = T
    else:
        first_move = np.argmax(moving_mask)
        grasp_idx = max(0, first_move - padding)
    indices = []
    curr = 0
    while curr < grasp_idx:
        indices.append(curr); curr += approach_stride
    start_dense = curr
    if indices and start_dense == indices[-1]: start_dense += 1
    for i in range(start_dense, T, 1): indices.append(i)
    return np.array(indices, dtype=int), grasp_idx

def pad_chunk(traj, start_idx, chunk_size):
    T = len(traj)
    end_idx = min(start_idx + chunk_size, T)
    actual_len = max(0, end_idx - start_idx)
    if actual_len == chunk_size: return traj[start_idx:end_idx]
    chunk = np.zeros((chunk_size, *traj.shape[1:]), dtype=traj.dtype)
    if actual_len > 0: chunk[:actual_len] = traj[start_idx:end_idx]
    last_frame = traj[-1]
    for p in range(actual_len, chunk_size): chunk[p] = last_frame
    return chunk

# ==========================================
# Image Loading Utils
# ==========================================
def _channels_for_mode(image_mode):
    """Returns channels per camera view: 1 for mask, 3 for masked_color or raw."""
    return 1 if image_mode == 'mask' else 3

def load_multi_view_mask(mask_dirs_dict, frame_idx, transform, camera_views, image_mode='mask'):
    """
    Loads images for multiple views and concatenates along the channel dimension.
    image_mode:
        'mask'         -> grayscale binary mask (1 ch per view)
        'masked_color' -> mask * raw RGB (3 ch per view)
        'raw'          -> raw RGB image (3 ch per view)
    """
    ch_per_view = _channels_for_mode(image_mode)
    tensors = []
    for view in camera_views:
        mdir = mask_dirs_dict[view]
        possible_names = [
            f"{frame_idx:04d}.png", f"{frame_idx + 1:04d}.png",
            f"{frame_idx}.png", f"{frame_idx + 1}.png",
            f"{frame_idx:05d}.png", f"{frame_idx:06d}.png",
            f"frame_{frame_idx:04d}.png"
        ]
        img_found = False
        for img_name in possible_names:
            img_path = Path(mdir) / img_name
            if img_path.exists():
                if image_mode == 'mask':
                    img = Image.open(img_path).convert('L')
                    tensors.append(transform(img))
                elif image_mode == 'masked_color':
                    mask_img = Image.open(img_path).convert('L')
                    raw_dir = str(mdir).replace('_frames_mask', '_frames')
                    raw_path = Path(raw_dir) / img_name
                    if raw_path.exists():
                        raw_img = Image.open(raw_path).convert('RGB')
                        mask_t = transform(mask_img)    # (1, 64, 64)
                        raw_t = transform(raw_img)      # (3, 64, 64)
                        tensors.append(mask_t * raw_t)  # (3, 64, 64) broadcast
                    else:
                        tensors.append(torch.zeros((3, 64, 64), dtype=torch.float32))
                elif image_mode == 'raw':
                    raw_dir = str(mdir).replace('_frames_mask', '_frames')
                    raw_path = Path(raw_dir) / img_name
                    if raw_path.exists():
                        raw_img = Image.open(raw_path).convert('RGB')
                        tensors.append(transform(raw_img))
                    else:
                        tensors.append(torch.zeros((3, 64, 64), dtype=torch.float32))
                img_found = True
                break
        if not img_found:
            tensors.append(torch.zeros((ch_per_view, 64, 64), dtype=torch.float32))
    return torch.cat(tensors, dim=0)

def pad_multi_view_image_chunk(mask_dirs_dict, frame_indices, start_idx, chunk_size, transform, camera_views, image_mode='mask'):
    T = len(frame_indices)
    end_idx = min(start_idx + chunk_size, T)
    actual_len = max(0, end_idx - start_idx)
    chunk_tensors = []
    for i in range(start_idx, end_idx):
        real_frame_idx = frame_indices[i]
        chunk_tensors.append(load_multi_view_mask(mask_dirs_dict, real_frame_idx, transform, camera_views, image_mode))
    if actual_len < chunk_size:
        ch_per_view = _channels_for_mode(image_mode)
        last_valid_tensor = chunk_tensors[-1] if chunk_tensors else torch.zeros((len(camera_views) * ch_per_view, 64, 64))
        for _ in range(actual_len, chunk_size):
            chunk_tensors.append(last_valid_tensor.clone())
    return torch.stack(chunk_tensors, dim=0)

# ==========================================
# Normalizer
# ==========================================
class Normalizer:
    def __init__(self, stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = None,
                 robot_state_type: str = 'joint', action_type: str = 'joint',
                 target_action_type: str = None):
        self.stats = stats if stats is not None else {}
        self.eps = 1e-6
        self.robot_state_type = robot_state_type
        self.action_type = action_type
        self.target_action_type = target_action_type

    def _norm_ee_pose(self, data, mean, std, is_tensor):
        """ee_idm: 14d=[l_pos,l_quat,r_pos,r_quat], stats are length-6 [l_pos_mean,r_pos_mean].
        Single-arm 7d falls through to the original logic."""
        last = data.shape[-1]
        if last == 7:
            pos = data[..., :3]; quat = data[..., 3:]
            pos_n = (pos - mean) / (std + self.eps)
            return torch.cat([pos_n, quat], dim=-1) if is_tensor else np.concatenate([pos_n, quat], axis=-1)
        # 14d dual-arm
        l_pos = data[..., 0:3]; l_quat = data[..., 3:7]
        r_pos = data[..., 7:10]; r_quat = data[..., 10:14]
        l_m, r_m = mean[..., :3], mean[..., 3:6]
        l_s, r_s = std[..., :3],  std[..., 3:6]
        l_n = (l_pos - l_m) / (l_s + self.eps)
        r_n = (r_pos - r_m) / (r_s + self.eps)
        cat = (lambda parts: torch.cat(parts, dim=-1)) if is_tensor else (lambda parts: np.concatenate(parts, axis=-1))
        return cat([l_n, l_quat, r_n, r_quat])

    def _norm_robot_pose_block(self, data, mean, std, is_tensor):
        """robot_state/action with type=ee. 10d=[pos,rot6d,grip]; 20d=[lpos,lrot6d,lgrip,rpos,rrot6d,rgrip]."""
        last = data.shape[-1]
        if last == 10:
            pos = data[..., :3]; rot6d = data[..., 3:9]; grip = data[..., 9:10]
            pos_m, pos_s = mean[..., :3], std[..., :3]
            grip_m, grip_s = mean[..., 9:10], std[..., 9:10]
            pos_n = (pos - pos_m) / (pos_s + self.eps)
            grip_n = (grip - grip_m) / (grip_s + self.eps)
            cat = (lambda parts: torch.cat(parts, dim=-1)) if is_tensor else (lambda parts: np.concatenate(parts, axis=-1))
            return cat([pos_n, rot6d, grip_n])
        # 20d dual-arm
        l_pos, l_r6, l_g = data[..., 0:3], data[..., 3:9], data[..., 9:10]
        r_pos, r_r6, r_g = data[..., 10:13], data[..., 13:19], data[..., 19:20]
        lpm, lps = mean[..., 0:3],  std[..., 0:3]
        lgm, lgs = mean[..., 9:10], std[..., 9:10]
        rpm, rps = mean[..., 10:13], std[..., 10:13]
        rgm, rgs = mean[..., 19:20], std[..., 19:20]
        lpn = (l_pos - lpm) / (lps + self.eps); lgn = (l_g - lgm) / (lgs + self.eps)
        rpn = (r_pos - rpm) / (rps + self.eps); rgn = (r_g - rgm) / (rgs + self.eps)
        cat = (lambda parts: torch.cat(parts, dim=-1)) if is_tensor else (lambda parts: np.concatenate(parts, axis=-1))
        return cat([lpn, l_r6, lgn, rpn, r_r6, rgn])

    def _unnorm_ee_pose(self, data, mean, std, is_tensor):
        last = data.shape[-1]
        if last == 7:
            pos = data[..., :3]; remain = data[..., 3:]
            pos_u = pos * (std + self.eps) + mean
            return torch.cat([pos_u, remain], dim=-1) if is_tensor else np.concatenate([pos_u, remain], axis=-1)
        l_pos, l_quat = data[..., 0:3], data[..., 3:7]
        r_pos, r_quat = data[..., 7:10], data[..., 10:14]
        l_m, r_m = mean[..., :3], mean[..., 3:6]
        l_s, r_s = std[..., :3], std[..., 3:6]
        l_u = l_pos * (l_s + self.eps) + l_m
        r_u = r_pos * (r_s + self.eps) + r_m
        cat = (lambda parts: torch.cat(parts, dim=-1)) if is_tensor else (lambda parts: np.concatenate(parts, axis=-1))
        return cat([l_u, l_quat, r_u, r_quat])

    def _unnorm_robot_pose_block(self, data, mean, std, is_tensor):
        last = data.shape[-1]
        if last == 10:
            pos, rot6d, grip = data[..., :3], data[..., 3:9], data[..., 9:10]
            pm, ps = mean[..., :3], std[..., :3]
            gm, gs = mean[..., 9:10], std[..., 9:10]
            pu = pos * (ps + self.eps) + pm; gu = grip * (gs + self.eps) + gm
            cat = (lambda parts: torch.cat(parts, dim=-1)) if is_tensor else (lambda parts: np.concatenate(parts, axis=-1))
            return cat([pu, rot6d, gu])
        l_pos, l_r6, l_g = data[..., 0:3], data[..., 3:9], data[..., 9:10]
        r_pos, r_r6, r_g = data[..., 10:13], data[..., 13:19], data[..., 19:20]
        lpm, lps = mean[..., 0:3], std[..., 0:3]
        lgm, lgs = mean[..., 9:10], std[..., 9:10]
        rpm, rps = mean[..., 10:13], std[..., 10:13]
        rgm, rgs = mean[..., 19:20], std[..., 19:20]
        lpu = l_pos * (lps + self.eps) + lpm; lgu = l_g * (lgs + self.eps) + lgm
        rpu = r_pos * (rps + self.eps) + rpm; rgu = r_g * (rgs + self.eps) + rgm
        cat = (lambda parts: torch.cat(parts, dim=-1)) if is_tensor else (lambda parts: np.concatenate(parts, axis=-1))
        return cat([lpu, l_r6, lgu, rpu, r_r6, rgu])

    def normalize(self, key, data):
        if key not in self.stats: return data
        mean, std = self.stats[key]
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            mean = torch.from_numpy(mean).to(device).float()
            std = torch.from_numpy(std).to(device).float()
        if key == 'ee_idm':
            return self._norm_ee_pose(data, mean, std, is_tensor)
        if (key in ('robot_state', 'robot_state_tp1')) and self.robot_state_type == 'ee':
            return self._norm_robot_pose_block(data, mean, std, is_tensor)
        is_ee_act    = (key == 'robot_action'        and self.action_type        == 'ee')
        is_ee_target = (key == 'robot_action_target' and self.target_action_type == 'ee')
        if is_ee_act or is_ee_target:
            return self._norm_robot_pose_block(data, mean, std, is_tensor)
        return (data - mean) / (std + self.eps)

    def unnormalize(self, key, data):
        if key not in self.stats: return data
        mean, std = self.stats[key]
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            mean = torch.from_numpy(mean).to(device).float()
            std = torch.from_numpy(std).to(device).float()
        if key == 'ee_idm':
            return self._unnorm_ee_pose(data, mean, std, is_tensor)
        if (key in ('robot_state', 'robot_state_tp1')) and self.robot_state_type == 'ee':
            return self._unnorm_robot_pose_block(data, mean, std, is_tensor)
        is_ee_act    = (key == 'robot_action'        and self.action_type        == 'ee')
        is_ee_target = (key == 'robot_action_target' and self.target_action_type == 'ee')
        if is_ee_act or is_ee_target:
            return self._unnorm_robot_pose_block(data, mean, std, is_tensor)
        return data * (std + self.eps) + mean

# ==========================================
# File Search
# ==========================================
def _list_npz(root_dirs: List[str], task_ids: Union[List[int], str] = 'all') -> List[Path]:
    files = []
    for rd in root_dirs:
        rd = Path(rd).resolve()
        if not rd.exists(): continue
        all_npz = sorted(rd.rglob("*.npz"))
        if task_ids == 'all': files.extend(all_npz)
        else:
            target_tasks = [f"task_{int(tid):02d}" for tid in task_ids]
            for path in all_npz:
                if any(t_str in path.parts for t_str in target_tasks): files.append(path)
    return files

# ==========================================
# UMI Dataset
# ==========================================
class UmiObjectDataset(Dataset):
    def __init__(self, root_dirs, subsample_stride=1, camera_views=None,
                 adaptive_subsample=True, max_episodes=-1, task_ids='all', chunk_size=1,
                 image_mode='mask'):
        super().__init__()
        if camera_views is None: camera_views = ['front']
        self.camera_views = camera_views
        self.chunk_size = chunk_size
        self.image_mode = image_mode
        self.files = sorted(_list_npz(root_dirs, task_ids=task_ids))
        if max_episodes >= 0: self.files = self.files[:max_episodes]
        self.samples = []; self._ee_trajs = []
        self._frame_indices = []; self._mask_dirs = []
        self.subsample_stride = subsample_stride
        self.adaptive_subsample = adaptive_subsample
        primary_view = self.camera_views[0]
        self.obj_key = f"blue_pose_{primary_view}"
        self.ee_key = f"umi_pose_{primary_view}"
        self.ee_key_alt = f"frank_ee_state_{primary_view}"   # dual-arm episodes
        self.trim_start = 0
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        print(f"[UMI] Tasks: {task_ids}. ImageMode: {image_mode}. Views: {self.camera_views}")

        for path in self.files:
            try:
                data = np.load(path, allow_pickle=True)
                if self.ee_key in data:
                    ee_traj = data[self.ee_key]
                elif self.ee_key_alt in data:
                    ee_traj = data[self.ee_key_alt]   # dual-arm: shape (T, 14)
                else:
                    ee_traj = None

                # Object pose is optional; it is only used to detect motion onset for subsampling
                obj_traj = data[self.obj_key] if self.obj_key in data else None

                if ee_traj is None:
                    traj_len = len(obj_traj) if obj_traj is not None else len(data.get("actions", []))
                    if traj_len == 0: continue
                    ee_traj = np.zeros((traj_len, 7), dtype=np.float32)

                if len(ee_traj) > self.trim_start:
                    ee_traj = ee_traj[self.trim_start:]
                    if obj_traj is not None: obj_traj = obj_traj[self.trim_start:]
                else: continue

                if self.adaptive_subsample:
                    # Without an object pose, detect motion onset from the EE trajectory
                    if obj_traj is not None:
                        indices, _ = get_adaptive_indices(obj_traj, self.subsample_stride, move_thresh=0.015)
                    else:
                        indices, _ = get_adaptive_indices(ee_traj, self.subsample_stride, move_thresh=0.005)
                else: indices = np.arange(0, len(ee_traj), self.subsample_stride)

                if len(indices) < 2: continue

                self._frame_indices.append(indices + self.trim_start)
                ee_traj = ee_traj[indices]
                self._ee_trajs.append(ee_traj)
                self._mask_dirs.append({v: path.parent / f"{v}_frames_mask" for v in self.camera_views})
                ep_idx = len(self._ee_trajs) - 1
                for t in range(len(ee_traj) - 1): self.samples.append((ep_idx, t))
            except Exception as e: print(f"[Error] UMI {path}: {e}")
        print(f"[UMI] Loaded {len(self._ee_trajs)} episodes.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        item = {"M": 0.0}
        d = self._mask_dirs[ep_idx]
        if self.chunk_size == 1:
            item["ee_t"] = self._ee_trajs[ep_idx][t]; item["ee_tp1"] = self._ee_trajs[ep_idx][t+1]
            item["img_mask_t"] = load_multi_view_mask(d, self._frame_indices[ep_idx][t], self.transform, self.camera_views, self.image_mode)
            item["img_mask_tp1"] = load_multi_view_mask(d, self._frame_indices[ep_idx][t+1], self.transform, self.camera_views, self.image_mode)
        else:
            item["ee_t"] = pad_chunk(self._ee_trajs[ep_idx], t, self.chunk_size)
            item["ee_tp1"] = pad_chunk(self._ee_trajs[ep_idx], t+1, self.chunk_size)
            item["img_mask_t"] = pad_multi_view_image_chunk(d, self._frame_indices[ep_idx], t, self.chunk_size, self.transform, self.camera_views, self.image_mode)
            item["img_mask_tp1"] = pad_multi_view_image_chunk(d, self._frame_indices[ep_idx], t+1, self.chunk_size, self.transform, self.camera_views, self.image_mode)
        return item

# ==========================================
# Frank (Sim) Dataset
# ==========================================
class FrankTransitionDataset(Dataset):
    def __init__(self, root_dirs, subsample_stride=1, split='train', val_ratio=0.1,
                 action_type='joint', robot_state_type='joint', camera_views=None,
                 task_ids='all', adaptive_subsample=True, max_episodes=-1,
                 target_action_type=None, chunk_size=1, image_mode='mask'):
        super().__init__()
        if camera_views is None: camera_views = ['front']
        self.camera_views = camera_views; self.chunk_size = chunk_size
        self.image_mode = image_mode
        all_files = sorted(_list_npz(root_dirs, task_ids))
        self.files = []
        if max_episodes >= 0:
            n_train = max_episodes
            n_val = max(1, int(n_train * val_ratio)) if (val_ratio > 0 and n_train > 0) else 0
            if split == 'train': self.files = all_files[:n_train]
            else: self.files = all_files[n_train:n_train + n_val]
        else:
            task_files_map = {}
            for f in all_files:
                task_key = "unknown"
                for part in f.parts:
                    if part.startswith("task_") and part[5:].isdigit(): task_key = part; break
                if task_key not in task_files_map: task_files_map[task_key] = []
                task_files_map[task_key].append(f)
            for task_name in sorted(task_files_map.keys()):
                t_files = sorted(task_files_map[task_name])
                n_val = max(1, int(len(t_files) * val_ratio)) if len(t_files) > 0 else 0
                n_train = len(t_files) - n_val
                if split == 'train': self.files.extend(t_files[:n_train])
                else: self.files.extend(t_files[n_train:])
        self.samples = []; self._state_trajs = []
        self._act_trajs = []; self._target_act_trajs = []; self._ee_idm_trajs = []
        self._frame_indices = []; self._mask_dirs = []
        self.robot_state_dim = None; self.robot_action_dim = None
        self.subsample_stride = subsample_stride; self.action_type = action_type
        self.target_action_type = target_action_type; self.robot_state_type = robot_state_type
        self.adaptive_subsample = adaptive_subsample
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        primary_view = self.camera_views[0]
        self.obj_key = f"blue_pose_{primary_view}"
        self.ee_state_key = f"frank_ee_state_{primary_view}"
        self.ee_action_key = f"frank_ee_action_executed_{primary_view}"
        self.trim_start = 0
        print(f"[Frank-{split}] Tasks: {task_ids}. Chunk: {self.chunk_size}. ImageMode: {image_mode}. Views: {self.camera_views}")

        for path in self.files:
            try:
                data = np.load(path, allow_pickle=True)
                joint_traj_full = data["joint_pos"]
                raw_actions_full = data["actions"]
                if len(raw_actions_full) == 0: continue

                # Object pose is optional; it is only used to detect motion onset for subsampling
                obj_traj_full = data[self.obj_key] if self.obj_key in data else None

                if self.ee_state_key in data: ee_idm_traj_full = data[self.ee_state_key]
                else:
                    if "ee_pose" in data and primary_view == 'world':
                        ee_idm_traj_full = data["ee_pose"][:, 7:14] if data["ee_pose"].shape[1] == 14 else np.zeros((len(raw_actions_full), 7))
                    else: ee_idm_traj_full = np.zeros((len(raw_actions_full), 7))

                if self.trim_start > 0:
                    if obj_traj_full is not None: obj_traj_full = obj_traj_full[self.trim_start:]
                    joint_traj_full = joint_traj_full[self.trim_start:]
                    raw_actions_full = raw_actions_full[self.trim_start:]
                    ee_idm_traj_full = ee_idm_traj_full[self.trim_start:]


                is_dual_arm = (ee_idm_traj_full.shape[-1] == 14)

                state_traj = None
                if self.robot_state_type == 'joint':
                    if is_dual_arm:
                        # dual-arm joint_pos = [l_arm(7), l_grip(1), r_arm(7), r_grip(1)] = 16
                        state_traj = joint_traj_full
                    else:
                        state_traj = joint_traj_full[:, 8:16] if joint_traj_full.shape[1] == 16 else joint_traj_full
                elif self.robot_state_type == 'ee':
                    if is_dual_arm:
                        l_pos, l_quat = ee_idm_traj_full[:, 0:3], ee_idm_traj_full[:, 3:7]
                        r_pos, r_quat = ee_idm_traj_full[:, 7:10], ee_idm_traj_full[:, 10:14]
                        l_rot6d = quat_to_6d_numpy(l_quat); r_rot6d = quat_to_6d_numpy(r_quat)
                        # dual-arm actions are 16-dim with grippers at index 7 (left) and 15 (right)
                        l_grip = raw_actions_full[:, 7:8]; r_grip = raw_actions_full[:, 15:16]
                        state_traj = np.concatenate([l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)  # (T, 20)
                    else:
                        pos = ee_idm_traj_full[:, :3]; quat = ee_idm_traj_full[:, 3:7]
                        rot6d = quat_to_6d_numpy(quat)
                        grip_idx = 7 if raw_actions_full.shape[1] == 8 else 15
                        grip = raw_actions_full[:, grip_idx:grip_idx+1]
                        state_traj = np.concatenate([pos, rot6d, grip], axis=-1)
                    
                T_len = min(len(state_traj), len(raw_actions_full), len(ee_idm_traj_full))
                if obj_traj_full is not None: T_len = min(T_len, len(obj_traj_full))

                if self.adaptive_subsample:
                    # Without an object pose, detect motion onset from the EE trajectory
                    if obj_traj_full is not None:
                        indices, _ = get_adaptive_indices(obj_traj_full[:T_len], self.subsample_stride, move_thresh=0.005)
                    else:
                        indices, _ = get_adaptive_indices(ee_idm_traj_full[:T_len], self.subsample_stride, move_thresh=0.005)
                else: indices = np.arange(0, T_len, self.subsample_stride)
                if len(indices) < 2: continue

                self._frame_indices.append(indices + self.trim_start)
                state_sub = state_traj[indices]; ee_idm_sub = ee_idm_traj_full[indices]
                ee_exec_full = None
                if self.ee_action_key in data: ee_exec_full = data[self.ee_action_key]
                elif "ee_pose" in data:
                    if data["ee_pose"].shape[1] == 14: ee_exec_full = data["ee_pose"][:, 7:14]
                if ee_exec_full is not None and self.trim_start > 0: ee_exec_full = ee_exec_full[self.trim_start:]
                
                def get_traj(atype):
                    n_cols = raw_actions_full.shape[1]
                    is_dual_arm = (ee_idm_traj_full.shape[-1] == 14)
                    if atype == 'joint':
                        if is_dual_arm:
                            # full 16-dim: [l_arm(7), l_grip(1), r_arm(7), r_grip(1)]
                            return raw_actions_full[indices, :]
                        if n_cols == 8:
                            j_sub = raw_actions_full[indices, 0:7]; grip_sub = raw_actions_full[indices, 7:8]
                        else:
                            j_sub = raw_actions_full[indices, 7:14]; grip_sub = raw_actions_full[indices, 15:16]
                        return np.concatenate([j_sub, grip_sub], axis=-1)
                    elif atype in ['ee', 'ee_delta']:
                        pose_src = ee_exec_full if ee_exec_full is not None else np.zeros((len(raw_actions_full), 14 if is_dual_arm else 7))
                        pose_sub = pose_src[indices]
                        if is_dual_arm:
                            l_pos, l_quat = pose_sub[:, 0:3], pose_sub[:, 3:7]
                            r_pos, r_quat = pose_sub[:, 7:10], pose_sub[:, 10:14]
                            l_rot6d, r_rot6d = quat_to_6d_numpy(l_quat), quat_to_6d_numpy(r_quat)
                            l_grip = raw_actions_full[indices, 7:8]; r_grip = raw_actions_full[indices, 15:16]
                            if atype == 'ee':
                                return np.concatenate([l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)  # (n, 20)
                            else:  # ee_delta
                                dl_p = np.zeros_like(l_pos); dl_p[:-1] = l_pos[1:] - l_pos[:-1]
                                dr_p = np.zeros_like(r_pos); dr_p[:-1] = r_pos[1:] - r_pos[:-1]
                                dl_r = np.zeros_like(l_rot6d); dl_r[:-1] = l_rot6d[1:] - l_rot6d[:-1]
                                dr_r = np.zeros_like(r_rot6d); dr_r[:-1] = r_rot6d[1:] - r_rot6d[:-1]
                                return np.concatenate([dl_p, dl_r, l_grip, dr_p, dr_r, r_grip], axis=-1)
                        # single-arm
                        grip_sub = raw_actions_full[indices, 7:8] if n_cols == 8 else raw_actions_full[indices, 15:16]
                        pos = pose_sub[:, :3]; quat = pose_sub[:, 3:7]; rot6d = quat_to_6d_numpy(quat)
                        if atype == 'ee':
                            return np.concatenate([pos, rot6d, grip_sub], axis=-1)
                        dp = np.zeros_like(pos); dp[:-1] = pos[1:] - pos[:-1]
                        dr = np.zeros_like(rot6d); dr[:-1] = rot6d[1:] - rot6d[:-1]
                        return np.concatenate([dp, dr, grip_sub], axis=-1)
                    return None
                    
                final_act = get_traj(self.action_type)
                final_target_act = get_traj(self.target_action_type) if self.target_action_type is not None else None
                if self.robot_state_dim is None:
                    self.robot_state_dim = state_sub.shape[-1]; self.robot_action_dim = final_act.shape[-1]
                self._state_trajs.append(state_sub)
                self._act_trajs.append(final_act)
                if final_target_act is not None: self._target_act_trajs.append(final_target_act)
                self._ee_idm_trajs.append(ee_idm_sub)
                self._mask_dirs.append({v: path.parent / f"{v}_frames_mask" for v in self.camera_views})
                ep_idx = len(self._state_trajs) - 1
                for t in range(len(state_sub) - 1): self.samples.append((ep_idx, t))
            except Exception as e: print(f"[Error] Frank {path}: {e}")
        print(f"[Frank-{split}] Loaded {len(self._state_trajs)} episodes.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        item = {"M": 1.0}
        d = self._mask_dirs[ep_idx]
        if self.chunk_size == 1:
            item.update({
                "robot_state": self._state_trajs[ep_idx][t], "robot_state_tp1": self._state_trajs[ep_idx][t+1],
                "robot_action": self._act_trajs[ep_idx][t], "ee_t": self._ee_idm_trajs[ep_idx][t],
                "ee_tp1": self._ee_idm_trajs[ep_idx][t+1]
            })
            if self._target_act_trajs: item["robot_action_target"] = self._target_act_trajs[ep_idx][t]
            item["img_mask_t"] = load_multi_view_mask(d, self._frame_indices[ep_idx][t], self.transform, self.camera_views, self.image_mode)
            item["img_mask_tp1"] = load_multi_view_mask(d, self._frame_indices[ep_idx][t+1], self.transform, self.camera_views, self.image_mode)
        else:
            item.update({
                "robot_state": pad_chunk(self._state_trajs[ep_idx], t, self.chunk_size),
                "robot_state_tp1": pad_chunk(self._state_trajs[ep_idx], t+1, self.chunk_size),
                "robot_action": pad_chunk(self._act_trajs[ep_idx], t, self.chunk_size),
                "ee_t": pad_chunk(self._ee_idm_trajs[ep_idx], t, self.chunk_size),
                "ee_tp1": pad_chunk(self._ee_idm_trajs[ep_idx], t+1, self.chunk_size)
            })
            if self._target_act_trajs: item["robot_action_target"] = pad_chunk(self._target_act_trajs[ep_idx], t, self.chunk_size)
            item["img_mask_t"] = pad_multi_view_image_chunk(d, self._frame_indices[ep_idx], t, self.chunk_size, self.transform, self.camera_views, self.image_mode)
            item["img_mask_tp1"] = pad_multi_view_image_chunk(d, self._frame_indices[ep_idx], t+1, self.chunk_size, self.transform, self.camera_views, self.image_mode)
        return item

# ==========================================
# Frank (Real) Dataset
# ==========================================
class FrankRealDataset(Dataset):
    def __init__(self, root_dirs, subsample_stride=1, split='train', val_ratio=0.1,
                 action_type='joint', robot_state_type='joint', camera_views=None,
                 task_ids='all', adaptive_subsample=True, max_episodes=-1,
                 target_action_type=None, chunk_size=1, image_mode='mask'):
        super().__init__()
        if camera_views is None: camera_views = ['front']
        self.camera_views = camera_views; self.chunk_size = chunk_size
        self.image_mode = image_mode
        all_files = sorted(_list_npz(root_dirs, task_ids))
        self.files = []
        if max_episodes >= 0:
            n_train = max_episodes
            n_val = max(1, int(n_train * val_ratio)) if (val_ratio > 0 and n_train > 0) else 0
            if split == 'train': self.files = all_files[:n_train]
            else: self.files = all_files[n_train:n_train + n_val]
        else:
            task_files_map = {}
            for f in all_files:
                task_key = "unknown"
                for part in f.parts:
                    if part.startswith("task_") and part[5:].isdigit(): task_key = part; break
                if task_key not in task_files_map: task_files_map[task_key] = []
                task_files_map[task_key].append(f)
            for task_name in sorted(task_files_map.keys()):
                t_files = sorted(task_files_map[task_name])
                n_val = max(1, int(len(t_files) * val_ratio)) if len(t_files) > 0 else 0
                n_train = len(t_files) - n_val
                if split == 'train': self.files.extend(t_files[:n_train])
                else: self.files.extend(t_files[n_train:])
        self.samples = []; self._state_trajs = []
        self._act_trajs = []; self._target_act_trajs = []; self._ee_idm_trajs = []
        self._frame_indices = []; self._mask_dirs = []
        self.robot_state_dim = None; self.robot_action_dim = None
        self.subsample_stride = subsample_stride; self.action_type = action_type
        self.target_action_type = target_action_type; self.robot_state_type = robot_state_type
        self.adaptive_subsample = adaptive_subsample
        self.transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])
        primary_view = self.camera_views[0]

        self.ee_state_key = f"frank_ee_state_{primary_view}"
        self.ee_action_key = f"frank_ee_action_executed_{primary_view}"
        self.trim_start = 0
        print(f"[FrankReal-{split}] Tasks: {task_ids}. Chunk: {self.chunk_size}. ImageMode: {image_mode}. Views: {self.camera_views}")
        for path in self.files:
            try:
                data = np.load(path, allow_pickle=True)
                joint_traj_full = data["joint_pos"]
                raw_actions_full = data["actions"]
                
                if self.ee_state_key in data: ee_idm_traj_full = data[self.ee_state_key]
                else:
                    if "ee_pose" in data and primary_view == 'world':
                        ee_idm_traj_full = data["ee_pose"][:, 7:14] if data["ee_pose"].shape[1] == 14 else np.zeros((len(raw_actions_full), 7))
                    else: ee_idm_traj_full = np.zeros((len(raw_actions_full), 7))
                
                if self.trim_start > 0:
                    joint_traj_full = joint_traj_full[self.trim_start:]
                    raw_actions_full = raw_actions_full[self.trim_start:]
                    ee_idm_traj_full = ee_idm_traj_full[self.trim_start:]
                is_dual_arm = (ee_idm_traj_full.shape[-1] == 14)

                state_traj = None
                if self.robot_state_type == 'joint':
                    if is_dual_arm:
                        # dual-arm joint_pos = [l_arm(7), l_grip(1), r_arm(7), r_grip(1)] = 16
                        state_traj = joint_traj_full
                    else:
                        state_traj = joint_traj_full[:, 8:16] if joint_traj_full.shape[1] == 16 else joint_traj_full
                elif self.robot_state_type == 'ee':
                    if is_dual_arm:
                        l_pos, l_quat = ee_idm_traj_full[:, 0:3], ee_idm_traj_full[:, 3:7]
                        r_pos, r_quat = ee_idm_traj_full[:, 7:10], ee_idm_traj_full[:, 10:14]
                        l_rot6d = quat_to_6d_numpy(l_quat); r_rot6d = quat_to_6d_numpy(r_quat)
                        # dual-arm actions are 16-dim with grippers at index 7 (left) and 15 (right)
                        l_grip = raw_actions_full[:, 7:8]; r_grip = raw_actions_full[:, 15:16]
                        state_traj = np.concatenate([l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)  # (T, 20)
                    else:
                        pos = ee_idm_traj_full[:, :3]; quat = ee_idm_traj_full[:, 3:7]
                        rot6d = quat_to_6d_numpy(quat)
                        grip_idx = 7 if raw_actions_full.shape[1] == 8 else 15
                        grip = raw_actions_full[:, grip_idx:grip_idx+1]
                        state_traj = np.concatenate([pos, rot6d, grip], axis=-1)
                
                T_len = min(len(state_traj), len(raw_actions_full), len(ee_idm_traj_full))
                
                if self.adaptive_subsample:
                    # Real data has no object pose, so detect motion onset from the EE trajectory
                    indices, _ = get_adaptive_indices(ee_idm_traj_full[:T_len], self.subsample_stride, move_thresh=0.005)
                else: indices = np.arange(0, T_len, self.subsample_stride)
                if len(indices) < 2: continue
                self._frame_indices.append(indices + self.trim_start)

                state_sub = state_traj[indices]; ee_idm_sub = ee_idm_traj_full[indices]
                ee_exec_full = None
                if self.ee_action_key in data: ee_exec_full = data[self.ee_action_key]
                elif "ee_pose" in data:
                    if data["ee_pose"].shape[1] == 14: ee_exec_full = data["ee_pose"][:, 7:14]
                if ee_exec_full is not None and self.trim_start > 0: ee_exec_full = ee_exec_full[self.trim_start:]
                
                def get_traj(atype):
                    n_cols = raw_actions_full.shape[1]
                    is_dual_arm = (ee_idm_traj_full.shape[-1] == 14)
                    if atype == 'joint':
                        if is_dual_arm:
                            # full 16-dim: [l_arm(7), l_grip(1), r_arm(7), r_grip(1)]
                            return raw_actions_full[indices, :]
                        if n_cols == 8:
                            j_sub = raw_actions_full[indices, 0:7]; grip_sub = raw_actions_full[indices, 7:8]
                        else:
                            j_sub = raw_actions_full[indices, 7:14]; grip_sub = raw_actions_full[indices, 15:16]
                        return np.concatenate([j_sub, grip_sub], axis=-1)
                    elif atype in ['ee', 'ee_delta']:
                        pose_src = ee_exec_full if ee_exec_full is not None else np.zeros((len(raw_actions_full), 14 if is_dual_arm else 7))
                        pose_sub = pose_src[indices]
                        if is_dual_arm:
                            l_pos, l_quat = pose_sub[:, 0:3], pose_sub[:, 3:7]
                            r_pos, r_quat = pose_sub[:, 7:10], pose_sub[:, 10:14]
                            l_rot6d, r_rot6d = quat_to_6d_numpy(l_quat), quat_to_6d_numpy(r_quat)
                            l_grip = raw_actions_full[indices, 7:8]; r_grip = raw_actions_full[indices, 15:16]
                            if atype == 'ee':
                                return np.concatenate([l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)  # (n, 20)
                            else:  # ee_delta
                                dl_p = np.zeros_like(l_pos); dl_p[:-1] = l_pos[1:] - l_pos[:-1]
                                dr_p = np.zeros_like(r_pos); dr_p[:-1] = r_pos[1:] - r_pos[:-1]
                                dl_r = np.zeros_like(l_rot6d); dl_r[:-1] = l_rot6d[1:] - l_rot6d[:-1]
                                dr_r = np.zeros_like(r_rot6d); dr_r[:-1] = r_rot6d[1:] - r_rot6d[:-1]
                                return np.concatenate([dl_p, dl_r, l_grip, dr_p, dr_r, r_grip], axis=-1)
                        # single-arm
                        grip_sub = raw_actions_full[indices, 7:8] if n_cols == 8 else raw_actions_full[indices, 15:16]
                        pos = pose_sub[:, :3]; quat = pose_sub[:, 3:7]; rot6d = quat_to_6d_numpy(quat)
                        if atype == 'ee':
                            return np.concatenate([pos, rot6d, grip_sub], axis=-1)
                        dp = np.zeros_like(pos); dp[:-1] = pos[1:] - pos[:-1]
                        dr = np.zeros_like(rot6d); dr[:-1] = rot6d[1:] - rot6d[:-1]
                        return np.concatenate([dp, dr, grip_sub], axis=-1)
                    return None
                
                final_act = get_traj(self.action_type)
                final_target_act = get_traj(self.target_action_type) if self.target_action_type is not None else None
                if self.robot_state_dim is None:
                    self.robot_state_dim = state_sub.shape[-1]; self.robot_action_dim = final_act.shape[-1]
                self._state_trajs.append(state_sub)
                self._act_trajs.append(final_act)
                if final_target_act is not None: self._target_act_trajs.append(final_target_act)
                self._ee_idm_trajs.append(ee_idm_sub)
                self._mask_dirs.append({v: path.parent / f"{v}_frames_mask" for v in self.camera_views})
                ep_idx = len(self._state_trajs) - 1
                for t in range(len(state_sub) - 1): self.samples.append((ep_idx, t))
            except Exception as e: print(f"[Error] FrankReal {path}: {e}")
        print(f"[FrankReal-{split}] Loaded {len(self._state_trajs)} episodes.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, t = self.samples[idx]
        item = {"M": 1.0}
        d = self._mask_dirs[ep_idx]
        if self.chunk_size == 1:
            item.update({
                "robot_state": self._state_trajs[ep_idx][t], "robot_state_tp1": self._state_trajs[ep_idx][t+1],
                "robot_action": self._act_trajs[ep_idx][t], "ee_t": self._ee_idm_trajs[ep_idx][t],
                "ee_tp1": self._ee_idm_trajs[ep_idx][t+1]
            })
            if self._target_act_trajs: item["robot_action_target"] = self._target_act_trajs[ep_idx][t]
            item["img_mask_t"] = load_multi_view_mask(d, self._frame_indices[ep_idx][t], self.transform, self.camera_views, self.image_mode)
            item["img_mask_tp1"] = load_multi_view_mask(d, self._frame_indices[ep_idx][t+1], self.transform, self.camera_views, self.image_mode)
        else:
            item.update({
                "robot_state": pad_chunk(self._state_trajs[ep_idx], t, self.chunk_size),
                "robot_state_tp1": pad_chunk(self._state_trajs[ep_idx], t+1, self.chunk_size),
                "robot_action": pad_chunk(self._act_trajs[ep_idx], t, self.chunk_size),
                "ee_t": pad_chunk(self._ee_idm_trajs[ep_idx], t, self.chunk_size),
                "ee_tp1": pad_chunk(self._ee_idm_trajs[ep_idx], t+1, self.chunk_size)
            })
            if self._target_act_trajs: item["robot_action_target"] = pad_chunk(self._target_act_trajs[ep_idx], t, self.chunk_size)
            item["img_mask_t"] = pad_multi_view_image_chunk(d, self._frame_indices[ep_idx], t, self.chunk_size, self.transform, self.camera_views, self.image_mode)
            item["img_mask_tp1"] = pad_multi_view_image_chunk(d, self._frame_indices[ep_idx], t+1, self.chunk_size, self.transform, self.camera_views, self.image_mode)
        return item

# ==========================================
# Combined Dataset
# ==========================================

class CombinedRoboticDataset(Dataset):
    def __init__(self, umi_roots, frank_roots, frank_real_roots=None, umi_subsample=1, frank_subsample=1,
                 split='train', val_ratio=0.1, action_type='joint', robot_state_type='joint',
                 camera_views=None, task_ids='all', adaptive_subsample=True,
                 frank_max_episodes=-1, umi_max_episodes=-1, frank_real_max_episodes=-1, 
                 umi_task_ids=None, frank_task_ids=None, frank_real_task_ids=None,
                 target_action_type=None, normalizer=None, chunk_size=1,
                 image_mode='mask'):
        if camera_views is None: camera_views = ['front']
        self.chunk_size = chunk_size; self.image_mode = image_mode
        final_umi_tasks = umi_task_ids if umi_task_ids is not None else task_ids
        final_frank_tasks = frank_task_ids if frank_task_ids is not None else task_ids
        final_frank_real_tasks = frank_real_task_ids if frank_real_task_ids is not None else task_ids

        if split == 'train':
            self.umi_ds = UmiObjectDataset(umi_roots, umi_subsample, camera_views,
                                           adaptive_subsample, umi_max_episodes, final_umi_tasks,
                                           chunk_size=self.chunk_size, image_mode=self.image_mode)
        else: self.umi_ds = []

        self.frank_ds = FrankTransitionDataset(frank_roots, frank_subsample, split, val_ratio,
                                               action_type, robot_state_type, camera_views,
                                               final_frank_tasks, adaptive_subsample, frank_max_episodes,
                                               target_action_type=target_action_type, chunk_size=self.chunk_size,
                                               image_mode=self.image_mode)

        if frank_real_roots is not None and len(frank_real_roots) > 0:
            self.frank_real_ds = FrankRealDataset(frank_real_roots, frank_subsample, split, val_ratio,
                                                  action_type, robot_state_type, camera_views,
                                                  final_frank_real_tasks, adaptive_subsample, frank_real_max_episodes,
                                                  target_action_type=target_action_type, chunk_size=self.chunk_size,
                                                  image_mode=self.image_mode)
        else: self.frank_real_ds = []
        
        if normalizer is not None:
            self.normalizer = normalizer
            self.normalizer.robot_state_type = robot_state_type
            self.normalizer.action_type = action_type
            self.normalizer.target_action_type = target_action_type
        else:
            self.normalizer = Normalizer(robot_state_type=robot_state_type, action_type=action_type, target_action_type=target_action_type)
            self.normalizer.stats = {}

        if 'robot_state' not in self.normalizer.stats:
            s_list, a_list, e_list = [], [], []
            if len(self.frank_ds._state_trajs) > 0:
                s_list.append(np.concatenate(self.frank_ds._state_trajs, axis=0))
                a_list.append(np.concatenate(self.frank_ds._act_trajs, axis=0))
                e_list.append(np.concatenate(self.frank_ds._ee_idm_trajs, axis=0))
            if hasattr(self, 'frank_real_ds') and len(self.frank_real_ds) > 0 and len(self.frank_real_ds._state_trajs) > 0:
                s_list.append(np.concatenate(self.frank_real_ds._state_trajs, axis=0))
                a_list.append(np.concatenate(self.frank_real_ds._act_trajs, axis=0))
                e_list.append(np.concatenate(self.frank_real_ds._ee_idm_trajs, axis=0))
                
            if s_list:
                s = np.concatenate(s_list, axis=0); a = np.concatenate(a_list, axis=0); e = np.concatenate(e_list, axis=0)
                self.normalizer.stats['robot_state'] = (np.mean(s, axis=0), np.std(s, axis=0))
                self.normalizer.stats['robot_action'] = (np.mean(a, axis=0), np.std(a, axis=0))
                if e.shape[-1] == 14:
                    l_pos = e[:, 0:3]; r_pos = e[:, 7:10]
                    self.normalizer.stats['ee_idm'] = (
                        np.concatenate([np.mean(l_pos, axis=0), np.mean(r_pos, axis=0)]),  # (6,)
                        np.concatenate([np.std (l_pos, axis=0), np.std (r_pos, axis=0)]),
                    )
                else:
                    self.normalizer.stats['ee_idm'] = (np.mean(e[:, :3], axis=0), np.std(e[:, :3], axis=0))
                
        # Infer target_action_dim from whichever dataset actually loaded target actions
        self.target_action_dim = None
        if target_action_type is not None:
            for ds in [self.frank_ds, getattr(self, 'frank_real_ds', None)]:
                if ds and hasattr(ds, '_target_act_trajs') and len(ds._target_act_trajs) > 0:
                    self.target_action_dim = ds._target_act_trajs[0].shape[-1]
                    if 'robot_action_target' not in self.normalizer.stats:
                        at = np.concatenate(ds._target_act_trajs, axis=0)
                        self.normalizer.stats['robot_action_target'] = (np.mean(at, axis=0), np.std(at, axis=0))
                    break
            if self.target_action_dim is None and 'robot_action_target' in self.normalizer.stats:
                self.target_action_dim = self.normalizer.stats['robot_action_target'][0].shape[0]
                
        # Take state/action dims from the first dataset that loaded successfully
        self.robot_state_dim = 8
        self.robot_action_dim = 8
        for ds in [self.frank_ds, getattr(self, 'frank_real_ds', None), self.umi_ds]:
            if ds and hasattr(ds, 'robot_state_dim') and getattr(ds, 'robot_state_dim') is not None:
                self.robot_state_dim = ds.robot_state_dim
                self.robot_action_dim = ds.robot_action_dim
                break
        # Detect dual-arm from any loaded Frank dataset's ee shape.
        self.is_dual_arm = False
        for ds in [self.frank_ds, getattr(self, 'frank_real_ds', None)]:
            if ds and hasattr(ds, '_ee_idm_trajs') and len(ds._ee_idm_trajs) > 0:
                self.is_dual_arm = (ds._ee_idm_trajs[0].shape[-1] == 14); break
        self.ee_internal_dim = 18 if self.is_dual_arm else 9
        self.umi_len = len(self.umi_ds); self.frank_len = len(self.frank_ds)
        self.frank_real_len = len(self.frank_real_ds) if hasattr(self, 'frank_real_ds') and self.frank_real_ds else 0
        self.total_len = self.umi_len + self.frank_len + self.frank_real_len
        self.target_action_type = target_action_type

    def __len__(self): return self.total_len

    def __getitem__(self, idx):
        # dataset_source: 0 = umi, 1 = frank_sim, 2 = frank_real
        if idx < self.umi_len:
            data = self.umi_ds[idx]
            dataset_source = 0
            if self.chunk_size == 1:
                data['robot_state'] = np.zeros(self.robot_state_dim, dtype=np.float32)
                data['robot_state_tp1'] = np.zeros(self.robot_state_dim, dtype=np.float32)
                data['robot_action'] = np.zeros(self.robot_action_dim, dtype=np.float32)
            else:
                data['robot_state'] = np.zeros((self.chunk_size, self.robot_state_dim), dtype=np.float32)
                data['robot_state_tp1'] = np.zeros((self.chunk_size, self.robot_state_dim), dtype=np.float32)
                data['robot_action'] = np.zeros((self.chunk_size, self.robot_action_dim), dtype=np.float32)
            if self.target_action_dim is not None:
                data['robot_action_target'] = np.zeros(self.target_action_dim, dtype=np.float32) if self.chunk_size == 1 else np.zeros((self.chunk_size, self.target_action_dim), dtype=np.float32)
        elif idx < self.umi_len + self.frank_len:
            data = self.frank_ds[idx - self.umi_len]
            dataset_source = 1
        else:
            data = self.frank_real_ds[idx - self.umi_len - self.frank_len]
            dataset_source = 2
            
        norm = self.normalizer
        ee_t = norm.normalize('ee_idm', data['ee_t']); ee_tp1 = norm.normalize('ee_idm', data['ee_tp1'])
        r_s = norm.normalize('robot_state', data['robot_state'])
        r_s_tp1 = norm.normalize('robot_state', data['robot_state_tp1'])
        r_a = norm.normalize('robot_action', data['robot_action'])
        item = {
            'ee_t': torch.from_numpy(ee_t).float(), 'ee_tp1': torch.from_numpy(ee_tp1).float(),
            'robot_state': torch.from_numpy(r_s).float(), 'robot_state_tp1': torch.from_numpy(r_s_tp1).float(),
            'robot_action': torch.from_numpy(r_a).float(), 'M': torch.tensor(data['M'], dtype=torch.float32),
            'dataset_source': torch.tensor(dataset_source, dtype=torch.long),
            'img_mask_t': data['img_mask_t'], 'img_mask_tp1': data['img_mask_tp1'],
        }
        if 'robot_action_target' in data:
            r_a_t = norm.normalize('robot_action_target', data['robot_action_target'])
            item['robot_action_target'] = torch.from_numpy(r_a_t).float()
        return item