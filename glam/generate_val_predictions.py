"""Generate action predictions on held-out episodes with a pretrained GLAM model.

For every episode, GLAM encodes each transition into an action latent (from the
IDM or the robot encoder) and a trained action decoder maps the latents to
actions. Predictions and ground truth are stored in a pkl file
(val_predictions_<source>.pkl) that sim_eval/eval_predictions_in_sim*.py can
replay in the MuJoCo Frank simulation.

Sources: 'frank' and 'frank_real' read the checkpoint's val_indices.txt; 'umi'
takes episode ids under --umi_base_dir (UMI has no robot actions, so its GT
actions are zeros and only the predictions are replayable).
"""
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from torchvision import transforms

from dataset import Normalizer, get_adaptive_indices, quat_to_6d_numpy, ensure_quat_continuity, load_multi_view_mask
from model import GLAM


class ActionDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        input_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


def integrate_delta_action(init_pose, deltas):
    """Integrate ee_delta predictions into absolute [pos(3), quat(4), grip(1)] actions."""
    T = len(deltas)
    abs_actions = np.zeros((T, 8))
    curr_pos = init_pose[:3]
    curr_quat = init_pose[3:]
    curr_6d = quat_to_6d_numpy(curr_quat[None, :])[0]

    for t in range(T):
        d_pos = deltas[t, :3]
        d_rot6d = deltas[t, 3:9]
        grip = deltas[t, 9:]

        curr_pos = curr_pos + d_pos
        curr_6d = curr_6d + d_rot6d

        a1 = curr_6d[:3]; a2 = curr_6d[3:]
        b1 = a1 / (np.linalg.norm(a1) + 1e-9)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-9)
        b3 = np.cross(b1, b2)
        mat = np.stack([b1, b2, b3], axis=1)
        q_new = R.from_matrix(mat).as_quat()

        abs_actions[t] = np.concatenate([curr_pos, q_new, grip])
    return abs_actions


def extract_action_sequence(data, indices, act_type, cam_view, trim_start, is_umi, umi_key_name=None, is_dual_arm=False):
    T_full = len(data["joint_pos"]) if not is_umi else len(data[umi_key_name])
    if is_umi: T_full -= trim_start

    if act_type == 'joint':
        if is_dual_arm:
            if is_umi:
                return np.zeros((len(indices), 16))
            # Dual-arm actions: [l_arm(7), l_grip(1), r_arm(7), r_grip(1)] = 16-dim
            raw = data["actions"][trim_start:]
            return raw[indices, :16]
        if is_umi: return np.zeros((len(indices), 8))
        raw = data["actions"][trim_start:]
        sub = raw[indices]
        n_cols = raw.shape[1]
        if n_cols == 8:
            j_sub = sub[:, 0:7]; g_sub = sub[:, 7:8]
        else:
            j_sub = sub[:, 7:14]; g_sub = sub[:, 15:16]
        return np.concatenate([j_sub, g_sub], axis=-1)

    elif act_type in ['ee', 'ee_delta']:
        if is_dual_arm:
            # Read 14-dim EE pose source (left+right pos+quat).
            if is_umi:
                # Dual-arm UMI stores EE under frank_ee_state_{view} (or executed_{view}); fallback to umi_pose
                ee_key_state = f"frank_ee_state_{cam_view}"
                ee_key_exec  = f"frank_ee_action_executed_{cam_view}"
                if ee_key_exec in data: pose_full = data[ee_key_exec][trim_start:]
                elif ee_key_state in data: pose_full = data[ee_key_state][trim_start:]
                elif umi_key_name in data: pose_full = data[umi_key_name][trim_start:]
                else: pose_full = np.zeros((T_full, 14))
                # Grippers from UMI actions: layout [l_q(6), r_q(6), l_grip(1), r_grip(1)] = 14
                if "actions" in data and data["actions"].shape[1] >= 14:
                    a = data["actions"][trim_start:]
                    l_grip_full = a[:, 12:13]; r_grip_full = a[:, 13:14]
                else:
                    l_grip_full = np.zeros((T_full, 1)); r_grip_full = np.zeros((T_full, 1))
            else:
                ee_key_exec = f"frank_ee_action_executed_{cam_view}"
                if ee_key_exec in data: pose_full = data[ee_key_exec][trim_start:]
                elif "ee_pose" in data and data["ee_pose"].shape[1] == 14:
                    pose_full = data["ee_pose"][trim_start:]
                else:
                    pose_full = np.zeros((T_full, 14))
                # Dual-arm Frank actions are ENV-layout: [l_arm(7), r_arm(7), l_grip(1), r_grip(1)] = 16
                if "actions" in data and data["actions"].shape[1] >= 16:
                    a = data["actions"][trim_start:]
                    l_grip_full = a[:, 14:15]; r_grip_full = a[:, 15:16]
                else:
                    l_grip_full = np.zeros((T_full, 1)); r_grip_full = np.zeros((T_full, 1))
            pose_sub = pose_full[indices]
            l_grip = l_grip_full[indices]; r_grip = r_grip_full[indices]
            l_pos, l_quat = pose_sub[:, :3], pose_sub[:, 3:7]
            r_pos, r_quat = pose_sub[:, 7:10], pose_sub[:, 10:14]
            l_quat = ensure_quat_continuity(l_quat); r_quat = ensure_quat_continuity(r_quat)
            l_rot6d = quat_to_6d_numpy(l_quat); r_rot6d = quat_to_6d_numpy(r_quat)
            if act_type == 'ee':
                return np.concatenate([l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)
            else:  # ee_delta
                dl_p = np.zeros_like(l_pos); dl_p[:-1] = l_pos[1:] - l_pos[:-1]
                dr_p = np.zeros_like(r_pos); dr_p[:-1] = r_pos[1:] - r_pos[:-1]
                dl_r = np.zeros_like(l_rot6d); dl_r[:-1] = l_rot6d[1:] - l_rot6d[:-1]
                dr_r = np.zeros_like(r_rot6d); dr_r[:-1] = r_rot6d[1:] - r_rot6d[:-1]
                return np.concatenate([dl_p, dl_r, l_grip, dr_p, dr_r, r_grip], axis=-1)

        if is_umi:
            raw_full = data[umi_key_name][trim_start:]
            grip = np.zeros((len(raw_full), 1))
            act_sub_raw = np.concatenate([raw_full, grip], axis=-1)[indices]
        else:
            if f"frank_ee_action_executed_{cam_view}" in data:
                ee_exec = data[f"frank_ee_action_executed_{cam_view}"][trim_start:]
            elif "ee_pose" in data:
                ee_exec = data["ee_pose"][trim_start:, 7:14] if data["ee_pose"].shape[1] == 14 else data["ee_pose"][trim_start:]
            else:
                ee_exec = np.zeros((len(data["actions"])-trim_start, 7))

            raw_a = data["actions"][trim_start:]
            n_cols = raw_a.shape[1]
            grip_idx = 7 if n_cols == 8 else 15
            grip = raw_a[:, grip_idx:grip_idx+1]

            act_sub_raw = np.concatenate([ee_exec, grip], axis=-1)[indices]

        pos = act_sub_raw[:, :3]; quat = act_sub_raw[:, 3:7]; grip = act_sub_raw[:, 7:]
        quat = ensure_quat_continuity(quat)
        rot6d = quat_to_6d_numpy(quat)

        if act_type == 'ee': return np.concatenate([pos, rot6d, grip], axis=-1)
        elif act_type == 'ee_delta':
            delta_pos = np.zeros_like(pos)
            delta_pos[:-1] = pos[1:] - pos[:-1]
            delta_rot = np.zeros_like(rot6d)
            delta_rot[:-1] = rot6d[1:] - rot6d[:-1]
            return np.concatenate([delta_pos, delta_rot, grip], axis=-1)
    return None


def load_and_prep_data(path, args, normalizer):
    try: data = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"[Error loading {path}]: {e}")
        return None

    cam_view = args.camera_views[0]
    obj_key = f"blue_pose_{cam_view}"

    # Object pose is optional; it is only used to detect motion onset for subsampling
    use_obj_for_adapt = True
    if obj_key in data:
        obj_traj = data[obj_key]
    else:
        if "joint_pos" in data: action_len = len(data["joint_pos"])
        elif f"umi_pose_{cam_view}" in data: action_len = len(data[f"umi_pose_{cam_view}"])
        else: return None
        if action_len == 0: return None
        obj_traj = np.zeros((action_len, 7), dtype=np.float32)
        obj_traj[:, 3] = 1.0  # identity quaternion
        use_obj_for_adapt = False

    trim_start = args.trim_start
    if len(obj_traj) <= trim_start: return None
    obj_traj = obj_traj[trim_start:]
    T_len = len(obj_traj)
    is_umi = (args.inference_source == 'umi')
    umi_key = f"umi_pose_{cam_view}"
    umi_dual_key = f"frank_ee_state_{cam_view}"  # dual-arm UMI stores EE under this key

    state_traj = None
    ee_idm_traj = None

    if not is_umi:
        joint_traj = data["joint_pos"][trim_start:]
        raw_actions = data["actions"][trim_start:]
        if f"frank_ee_state_{cam_view}" in data:
            ee_idm_traj = data[f"frank_ee_state_{cam_view}"][trim_start:]
        elif "ee_pose" in data and data["ee_pose"].shape[1] >= 14:
            ee_idm_traj = data["ee_pose"][trim_start:, 7:14]
        else:
            ee_idm_traj = np.zeros((T_len, 7))

        is_dual_arm = (ee_idm_traj.shape[-1] == 14) or (raw_actions.shape[-1] >= 16 and raw_actions.shape[-1] != 8)

        if args.robot_state_type == 'joint':
            if is_dual_arm:
                # Dual-arm joint_pos = [l_arm(7), l_grip(1), r_arm(7), r_grip(1)] = 16
                state_traj = joint_traj
            else:
                state_traj = joint_traj[:, 8:16] if joint_traj.shape[1] == 16 else joint_traj
        elif args.robot_state_type == 'ee':
            if is_dual_arm:
                l_pos = ee_idm_traj[:, 0:3]; l_quat = ee_idm_traj[:, 3:7]
                r_pos = ee_idm_traj[:, 7:10]; r_quat = ee_idm_traj[:, 10:14]
                l_quat = ensure_quat_continuity(l_quat); r_quat = ensure_quat_continuity(r_quat)
                l_rot6d = quat_to_6d_numpy(l_quat); r_rot6d = quat_to_6d_numpy(r_quat)
                # Dual-arm Frank actions are ENV-layout: [l_arm(7), r_arm(7), l_grip(1), r_grip(1)]
                if raw_actions.shape[1] >= 16:
                    l_grip = raw_actions[:, 14:15]; r_grip = raw_actions[:, 15:16]
                else:
                    l_grip = np.zeros((T_len, 1)); r_grip = np.zeros((T_len, 1))
                state_traj = np.concatenate([l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)
            else:
                pos = ee_idm_traj[:, :3]; quat = ee_idm_traj[:, 3:7]
                quat = ensure_quat_continuity(quat); rot6d = quat_to_6d_numpy(quat)
                n_cols = raw_actions.shape[1]
                grip_idx = 7 if n_cols == 8 else 15
                grip = raw_actions[:, grip_idx:grip_idx+1]
                state_traj = np.concatenate([pos, rot6d, grip], axis=-1)

        init_qpos_full = data["joint_pos"][0]
        init_ee_pose = ee_idm_traj[0]
    else:
        # UMI: prefer single-arm umi_pose; if missing, try the dual-arm frank_ee_state key.
        if umi_key in data:
            ee_idm_traj = data[umi_key][trim_start:]
        elif umi_dual_key in data:
            ee_idm_traj = data[umi_dual_key][trim_start:]
        else:
            ee_idm_traj = np.zeros((T_len, 7))
        is_dual_arm = (ee_idm_traj.shape[-1] == 14)
        state_dim = 8
        if 'robot_state' in normalizer.stats: state_dim = normalizer.stats['robot_state'][0].shape[0]
        state_traj = np.zeros((T_len, state_dim))
        if is_dual_arm:
            init_qpos_full = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04,
                                       0,  0.785, 0,  2.356, 0, 1.571, 0.785, 0.04])  # dual home (16-d, dataset layout)
        else:
            init_qpos_full = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
        init_ee_pose = ee_idm_traj[0] if len(ee_idm_traj) > 0 else np.zeros(7)

    subsample_stride = args.umi_subsample if is_umi else args.frank_subsample
    if args.adaptive_subsample:
        thresh = 0.015 if is_umi else 0.005
        if use_obj_for_adapt:
            indices, _ = get_adaptive_indices(obj_traj, subsample_stride, move_thresh=thresh)
        else:
            indices, _ = get_adaptive_indices(ee_idm_traj, subsample_stride, move_thresh=thresh)
    else:
        indices = np.arange(0, T_len, subsample_stride)

    if len(indices) < args.chunk_size + 1: return None

    obj_sub = obj_traj[indices]
    state_sub = state_traj[indices]
    ee_idm_sub = ee_idm_traj[indices]
    # For dual-arm UMI, point extract_action_sequence at the right key
    umi_key_for_action = umi_dual_key if (is_umi and is_dual_arm and umi_dual_key in data) else umi_key
    input_act_sub = extract_action_sequence(data, indices, args.action_type, cam_view, trim_start, is_umi, umi_key_for_action, is_dual_arm=is_dual_arm)
    target_gt_sub = extract_action_sequence(data, indices, args.target_action_type, cam_view, trim_start, is_umi, umi_key_for_action, is_dual_arm=is_dual_arm)

    if args.target_action_type in ['ee', 'ee_delta'] and not is_dual_arm:
        if is_umi:
            raw_pose = data[umi_key][trim_start:][indices[0]]
            init_ee_pose = np.concatenate([raw_pose, [0.0]])
        else:
            if f"frank_ee_action_executed_{cam_view}" in data: raw_pose = data[f"frank_ee_action_executed_{cam_view}"][trim_start:][indices[0]]
            elif "ee_pose" in data: raw_pose = data["ee_pose"][trim_start:, 7:14][indices[0]] if data["ee_pose"].shape[1] == 14 else data["ee_pose"][trim_start:][indices[0]]
            else: raw_pose = np.zeros(7)
            raw_act = data["actions"][trim_start:][indices[0]]
            n_cols = raw_act.shape[0]
            grip_idx = 7 if n_cols == 8 else 15
            grip = raw_act[grip_idx:grip_idx+1]
            init_ee_pose = np.concatenate([raw_pose, grip])
    # Note: dual-arm + ee_delta integration is not supported (use a joint target for dual-arm tasks).

    image_mode = 'mask'
    if getattr(args, 'masked_color_image', False): image_mode = 'masked_color'
    elif getattr(args, 'raw_image', False): image_mode = 'raw'

    mask_dirs_dict = {v: Path(path).parent / f"{v}_frames_mask" for v in args.camera_views}
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    imgs = []
    for idx in indices:
        real_idx = idx + trim_start
        imgs.append(load_multi_view_mask(mask_dirs_dict, real_idx, transform, args.camera_views, image_mode=image_mode))
    img_mask_sub = torch.stack(imgs)

    return {
        'obj': obj_sub, 'state': state_sub, 'input_action': input_act_sub,
        'target_action': target_gt_sub, 'ee_idm': ee_idm_sub,
        'init_qpos': init_qpos_full, 'init_ee_pose': init_ee_pose,
        'indices': indices, 'img_mask': img_mask_sub
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading GLAM: {args.glam_ckpt}")
    glam_ckpt = torch.load(args.glam_ckpt, map_location=device)
    if 'args' in glam_ckpt:
        saved_args = glam_ckpt['args']
        input_keys = [
            'latent_dim', 'robot_state_type', 'action_type',
            'forward_robot', 'use_idm_ee', 'use_idm_ee_part',
            'idm_obj_only', 'strict_robot_encoder',
            'recon_action', 'recon_action_rob', 'rob_latent_forward',
            'robot_as_posterior', 'camera_views', 'chunk_size',
            'masked_color_image', 'raw_image'
        ]
        for k in input_keys:
            if k in saved_args: setattr(args, k, saved_args[k])
            else:
                if k in ['masked_color_image', 'raw_image']: setattr(args, k, False)

    image_mode = 'mask'
    if getattr(args, 'masked_color_image', False): image_mode = 'masked_color'
    elif getattr(args, 'raw_image', False): image_mode = 'raw'

    K = args.chunk_size
    glam_run_name = Path(args.glam_ckpt).parent.name
    final_out_dir = Path(args.out_dir) / glam_run_name
    final_out_dir.mkdir(parents=True, exist_ok=True)

    stats = glam_ckpt['stats']
    normalizer = Normalizer(stats, robot_state_type=args.robot_state_type)
    s_dim = stats['robot_state'][0].shape[0]
    a_dim = stats['robot_action'][0].shape[0]
    ee_internal_dim = 18 if (stats.get('ee_idm') is not None and stats['ee_idm'][0].shape[0] == 6) else 9
    print(f"[CONFIG] s_dim={s_dim}, a_dim={a_dim}, ee_internal_dim={ee_internal_dim}")

    glam = GLAM(
        robot_state_dim=s_dim, robot_action_dim=a_dim, action_latent_dim=args.latent_dim,
        use_robot_forward=args.forward_robot, use_idm_ee=args.use_idm_ee,
        use_idm_ee_part=args.use_idm_ee_part, idm_obj_only=getattr(args, 'idm_obj_only', False),
        strict_robot_encoder=args.strict_robot_encoder, use_action_recon=args.recon_action,
        use_rob_action_recon=getattr(args, 'recon_action_rob', False),
        rob_latent_forward=getattr(args, 'rob_latent_forward', False), chunk_size=K,
        num_cameras=len(args.camera_views), image_mode=image_mode,
        ee_internal_dim=ee_internal_dim
    ).to(device)
    glam.load_state_dict(glam_ckpt['model_state'])
    glam.eval()

    dec_ckpt = torch.load(args.decoder_ckpt, map_location=device)
    if 'args' in dec_ckpt and 'target_action_type' in dec_ckpt['args']:
        args.target_action_type = dec_ckpt['args']['target_action_type']
    else:
        if not hasattr(args, 'target_action_type') or args.target_action_type is None:
            args.target_action_type = args.action_type

    if 'stats' in dec_ckpt: normalizer.stats.update(dec_ckpt['stats'])
    target_dim_size = a_dim
    if 'robot_action_target' in normalizer.stats:
        target_dim_size = normalizer.stats['robot_action_target'][0].shape[0]

    decoder = ActionDecoder(args.latent_dim, target_dim_size, hidden_dim=256, num_layers=4).to(device)
    decoder.load_state_dict(dec_ckpt['decoder_state'])
    decoder.eval()

    file_list = []
    if args.inference_source == 'umi':
        if args.manual_files: file_list = args.manual_files
        elif args.umi_base_dir and args.episode_ids:
            base_path = Path(args.umi_base_dir)
            for ep_id in args.episode_ids:
                file_path = base_path / f"{ep_id:04d}" / "umi_blue_grasp.npz"
                if file_path.exists(): file_list.append(str(file_path))
        else: raise ValueError("Provide manual_files or umi_base_dir + episode_ids")
    elif args.inference_source == 'frank':
        val_idx_path = Path(args.glam_ckpt).parent / "val_indices.txt"
        with open(val_idx_path, 'r') as f: all_files = [line.strip() for line in f if line.strip()]
        file_list = [f for f in all_files if "frank_real" not in f and "frank" in f][:args.num_episodes]
    elif args.inference_source == 'frank_real':
        val_idx_path = Path(args.glam_ckpt).parent / "val_indices.txt"
        with open(val_idx_path, 'r') as f: all_files = [line.strip() for line in f if line.strip()]
        file_list = [f for f in all_files if "frank_real" in f][:args.num_episodes]

    results_db = {}

    for ep_path in tqdm(file_list, desc=f"Inference ({args.inference_source})"):
        raw_data = load_and_prep_data(ep_path, args, normalizer)
        if raw_data is None: continue

        T_len = len(raw_data['obj'])
        valid_len = T_len - K
        if valid_len <= 0: continue

        rob_s_list, rob_a_list = [], []
        ee_t_list, ee_tp1_list = [], []
        target_a_list = []
        img_t_list, img_tp1_list = [], []

        for t in range(valid_len):
            rob_s_list.append(raw_data['state'][t : t+K])
            rob_a_list.append(raw_data['input_action'][t : t+K])
            ee_t_list.append(raw_data['ee_idm'][t : t+K])
            ee_tp1_list.append(raw_data['ee_idm'][t+1 : t+1+K])
            target_a_list.append(raw_data['target_action'][t : t+K])
            img_t_list.append(raw_data['img_mask'][t : t+K])
            img_tp1_list.append(raw_data['img_mask'][t+1 : t+1+K])

        rob_s = torch.from_numpy(np.array(rob_s_list)).float().to(device)
        rob_a = torch.from_numpy(np.array(rob_a_list)).float().to(device)
        ee_t = torch.from_numpy(np.array(ee_t_list)).float().to(device)
        ee_tp1 = torch.from_numpy(np.array(ee_tp1_list)).float().to(device)
        target_a = np.array(target_a_list)
        img_t_tensor = torch.stack(img_t_list).to(device)
        img_tp1_tensor = torch.stack(img_tp1_list).to(device)
        if K == 1:
            rob_s = rob_s.squeeze(1); rob_a = rob_a.squeeze(1)
            ee_t = ee_t.squeeze(1); ee_tp1 = ee_tp1.squeeze(1)
            target_a = target_a.squeeze(1)
            img_t_tensor = img_t_tensor.squeeze(1)
            img_tp1_tensor = img_tp1_tensor.squeeze(1)

        rob_s_n = normalizer.normalize('robot_state', rob_s)
        rob_a_n = normalizer.normalize('robot_action', rob_a)
        ee_t_n = normalizer.normalize('ee_idm', ee_t)
        ee_tp1_n = normalizer.normalize('ee_idm', ee_tp1)

        B = img_t_tensor.shape[0]

        with torch.no_grad():
            if args.encoder_source == 'robot':
                if K > 1:
                    enc_in = torch.cat([rob_s_n.view(B*K, -1), rob_a_n.view(B*K, -1)], dim=-1)
                else:
                    enc_in = torch.cat([rob_s_n, rob_a_n], dim=-1)
                mu_flat, _ = glam.robot_enc(enc_in)
                mu = mu_flat.view(B, K, -1) if K > 1 else mu_flat

            elif args.encoder_source == 'idm':
                obj_t_feat = glam.encode_vision(img_t_tensor)
                obj_tp1_feat = glam.encode_vision(img_tp1_tensor)
                obj_t_idm = obj_t_feat[:, 0] if K > 1 else obj_t_feat
                obj_tp1_idm = obj_tp1_feat[:, -1] if K > 1 else obj_tp1_feat

                if args.use_idm_ee:
                    ee_t_9d = glam.process_pose_input(ee_t_n)
                    ee_t_idm = ee_t_9d[:, 0] if K > 1 else ee_t_9d
                    if getattr(glam, "use_idm_ee_part", False):
                        idm_in = torch.cat([obj_t_idm, ee_t_idm, obj_tp1_idm], dim=-1)
                    else:
                        ee_tp1_9d = glam.process_pose_input(ee_tp1_n)
                        ee_tp1_idm = ee_tp1_9d[:, -1] if K > 1 else ee_tp1_9d
                        idm_in = torch.cat([obj_t_idm, ee_t_idm, obj_tp1_idm, ee_tp1_idm], dim=-1)
                else:
                    idm_in = torch.cat([obj_t_idm, obj_tp1_idm], dim=-1)

                mu_flat, _ = glam.idm(idm_in)
                mu = mu_flat.view(B, K, -1) if K > 1 else mu_flat

            if K > 1:
                mu_for_dec = mu.view(B*K, -1)
                pred_action_n_flat = decoder(mu_for_dec)
                pred_action_n = pred_action_n_flat.view(B, K, -1)
            else:
                pred_action_n = decoder(mu)

            key = 'robot_action_target' if 'robot_action_target' in normalizer.stats else 'robot_action'
            pred_action = normalizer.unnormalize(key, pred_action_n).cpu().numpy()

        pred_action_abs = None
        if args.target_action_type == 'ee_delta':
            init_pose = raw_data['init_ee_pose']
            pred_for_int = pred_action[:, 0, :] if K > 1 else pred_action
            pred_action_abs = integrate_delta_action(init_pose, pred_for_int)

        try:
            ep_id_str = Path(ep_path).parent.name
            db_key = f"UMI_Transfer_Ep{ep_id_str}" if ep_id_str.isdigit() else f"UMI_Transfer_{Path(ep_path).stem}"
        except Exception:
            db_key = f"UMI_Transfer_{Path(ep_path).stem}"

        if args.inference_source != 'umi':
            db_key = f"{Path(ep_path).parent.parent.name}_{Path(ep_path).parent.name}"

        results_db[db_key] = {
            'pred_action': pred_action,
            'pred_action_abs': pred_action_abs,
            'gt_action': target_a,
            'init_qpos': raw_data['init_qpos'],
            'action_type': args.target_action_type,
            'source_type': args.inference_source,
            'umi_file_path': str(ep_path),
            # Original-episode frame index for each prediction step (after
            # subsampling/trim), so replays can align source video frames.
            'frame_indices': (np.asarray(raw_data['indices'][:valid_len]) + args.trim_start).astype(np.int64),
        }

    out_name = f"val_predictions_{args.inference_source}.pkl"
    save_path = final_out_dir / out_name
    with open(save_path, 'wb') as f: pickle.dump(results_db, f)
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glam_ckpt", type=str, required=True)
    parser.add_argument("--decoder_ckpt", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="inference_results")
    parser.add_argument("--encoder_source", type=str, default="robot", choices=['robot', 'idm'])

    parser.add_argument("--frank_subsample", type=int, default=1)
    parser.add_argument("--umi_subsample", type=int, default=2)
    parser.add_argument("--trim_start", type=int, default=0)
    parser.add_argument("--adaptive_subsample", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=1)

    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--inference_source", type=str, default="frank", choices=["frank", "umi", "frank_real"])
    parser.add_argument("--manual_files", nargs='+', type=str)
    parser.add_argument("--umi_base_dir", type=str)
    parser.add_argument("--episode_ids", nargs='+', type=int)

    # Model configuration is synced from the GLAM checkpoint at startup;
    # these flags only act as fallbacks for checkpoints without saved args.
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--robot_state_type", type=str, default="joint")
    parser.add_argument("--action_type", type=str, default="joint")
    parser.add_argument("--target_action_type", type=str, default=None)
    parser.add_argument("--camera_views", nargs='+', default=["front"], help="List of views e.g. front overhead")
    parser.add_argument("--forward_robot", action="store_true")
    parser.add_argument("--use_idm_ee", action="store_true")
    parser.add_argument("--use_idm_ee_part", action="store_true")
    parser.add_argument("--idm_obj_only", action="store_true")
    parser.add_argument("--recon_action_rob", action="store_true")
    parser.add_argument("--rob_latent_forward", action="store_true")
    parser.add_argument("--strict_robot_encoder", action="store_true")
    parser.add_argument("--recon_action", action="store_true")
    parser.add_argument("--robot_as_posterior", action="store_true")
    parser.add_argument("--masked_color_image", action="store_true")
    parser.add_argument("--raw_image", action="store_true")

    args = parser.parse_args()
    main(args)
