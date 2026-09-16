import argparse
import torch
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import shutil
from torchvision import transforms

from dataset import Normalizer, get_adaptive_indices, quat_to_6d_numpy, load_multi_view_mask
from model import GLAM

FIXED_TARGET_POS = np.array([0.40, -0.05, 0.90], dtype=np.float32)

def load_and_prep_file(file_path, args, normalizer, data_type="frank"):
    """
    data_type: "frank", "frank_real", or "umi"
    """
    is_umi = (data_type == "umi")
    try: data = np.load(file_path, allow_pickle=True)
    except Exception as e: print(f"[Error] Failed to load {file_path}: {e}"); return None

    cam_view = args.camera_views[0] if hasattr(args, 'camera_views') and args.camera_views else args.camera_view
    obj_key = f"blue_pose_{cam_view}"

    # Object pose may be missing (e.g. real data); fall back to a dummy pose.
    # obj_raw is still saved so the policy dataset keeps a consistent format.
    use_obj_for_adapt = True
    if obj_key in data:
        obj_traj = data[obj_key]
    else:
        if "joint_pos" in data: action_len = len(data["joint_pos"])
        elif f"umi_pose_{cam_view}" in data: action_len = len(data[f"umi_pose_{cam_view}"])
        elif "actions" in data: action_len = len(data["actions"])
        else: return None

        if action_len == 0: return None
        obj_traj = np.zeros((action_len, 7), dtype=np.float32)
        obj_traj[:, 3] = 1.0  # identity quaternion
        use_obj_for_adapt = False

    trim_start = 0
    # Detect dual-arm from the normalizer stats
    is_dual_arm = (normalizer.stats.get('ee_idm') is not None and
                   normalizer.stats['ee_idm'][0].shape[0] == 6)

    if is_umi:
        # Dual-arm UMI episodes store EE as frank_ee_state_{view} (14d); single-arm uses umi_pose_{view} (7d).
        umi_ee_key = f"umi_pose_{cam_view}"
        dual_ee_key = f"frank_ee_state_{cam_view}"
        if dual_ee_key in data:
            ee_idm_traj = data[dual_ee_key]
        elif umi_ee_key in data:
            ee_idm_traj = data[umi_ee_key]
        else:
            ee_idm_traj = np.zeros((len(obj_traj), 14 if is_dual_arm else 7))
        joint_dim = 16 if is_dual_arm else 8
        joint_traj = np.zeros((len(obj_traj), joint_dim), dtype=np.float32)
    else:
        ee_key = f"frank_ee_state_{cam_view}"
        if ee_key in data:
            ee_idm_traj = data[ee_key]
        elif "ee_pose" in data and data["ee_pose"].shape[1] >= 14:
            ee_idm_traj = data["ee_pose"][:, 7:14]
        else:
            ee_idm_traj = np.zeros((len(obj_traj), 14 if is_dual_arm else 7))

        if 'joint_pos' in data:
            raw_j = data['joint_pos']
            if is_dual_arm:
                # dual-arm joint_pos = [l_arm(7), l_grip(1), r_arm(7), r_grip(1)] = 16
                joint_traj = raw_j if raw_j.shape[1] == 16 else np.zeros((len(obj_traj), 16), dtype=np.float32)
            else:
                if raw_j.shape[1] == 16:
                    joint_traj = np.concatenate([raw_j[:, 7:14], raw_j[:, 15:16]], axis=-1)
                elif raw_j.shape[1] == 8:
                    joint_traj = raw_j
                else:
                    joint_traj = np.zeros((len(obj_traj), 8), dtype=np.float32)
        else:
            joint_traj = np.zeros((len(obj_traj), 16 if is_dual_arm else 8), dtype=np.float32)

    # Target position for the saved obs_target: the orange cube's initial pose
    # (both stacking tasks place onto orange); fixed fallback if missing.
    if "orange_pose_world_init" in data:
        target_pos = data["orange_pose_world_init"][:3].astype(np.float32)
    else:
        target_pos = FIXED_TARGET_POS.copy()
    target_traj = np.tile(target_pos, (len(obj_traj), 1)).astype(np.float32)

    real_action_for_save = None; action_ee_for_save = None
    if is_umi:
        # action_real for UMI: zeros at the right dim (downstream policy can mask it out).
        act_dim = 8
        if 'robot_action' in normalizer.stats:
            act_dim = normalizer.stats['robot_action'][0].shape[0]
        real_action_for_save = np.zeros((len(obj_traj), act_dim))

        if is_dual_arm:
            # Build 20-d ee action from 14-d ee_idm_traj. Grippers from UMI actions if present
            # (dual-arm UMI actions layout: [l_q(6), r_q(6), l_grip(1), r_grip(1)] = 14).
            l_pos = ee_idm_traj[:, 0:3]; l_quat = ee_idm_traj[:, 3:7]
            r_pos = ee_idm_traj[:, 7:10]; r_quat = ee_idm_traj[:, 10:14]
            l_rot6d = quat_to_6d_numpy(l_quat); r_rot6d = quat_to_6d_numpy(r_quat)
            if "actions" in data and data["actions"].shape[1] >= 14:
                a = data["actions"]
                l_grip = a[:, 12:13]; r_grip = a[:, 13:14]
            else:
                l_grip = np.zeros((len(ee_idm_traj), 1)); r_grip = np.zeros((len(ee_idm_traj), 1))
            action_ee_for_save = np.concatenate(
                [l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)  # (T, 20)
        else:
            pos = ee_idm_traj[:, :3]; quat = ee_idm_traj[:, 3:7]
            rot6d = quat_to_6d_numpy(quat); grip = np.zeros((len(ee_idm_traj), 1))
            action_ee_for_save = np.concatenate([pos, rot6d, grip], axis=-1)  # (T, 10)
    else:
        raw_actions = data["actions"]
        if is_dual_arm:
            # Keep full 16-d dual-arm action: [l_arm(7), l_grip(1), r_arm(7), r_grip(1)]
            real_action_for_save = raw_actions if raw_actions.shape[1] == 16 else np.zeros((len(obj_traj), 16))
        else:
            if raw_actions.shape[1] == 16:
                real_action_for_save = np.concatenate([raw_actions[:, 7:14], raw_actions[:, 15:16]], axis=-1)
            else:
                real_action_for_save = raw_actions

        ee_exec_key = f"frank_ee_action_executed_{cam_view}"
        if ee_exec_key in data: ee_exec = data[ee_exec_key]
        elif "ee_pose" in data and data["ee_pose"].shape[1] >= 14: ee_exec = data["ee_pose"][:, 7:14] if not is_dual_arm else data["ee_pose"]
        else: ee_exec = np.zeros((len(obj_traj), 14 if is_dual_arm else 7))

        if is_dual_arm:
            l_pos = ee_exec[:, 0:3]; l_quat = ee_exec[:, 3:7]
            r_pos = ee_exec[:, 7:10]; r_quat = ee_exec[:, 10:14]
            l_rot6d = quat_to_6d_numpy(l_quat); r_rot6d = quat_to_6d_numpy(r_quat)
            if raw_actions.shape[1] >= 16:
                l_grip = raw_actions[:, 7:8]; r_grip = raw_actions[:, 15:16]
            else:
                l_grip = np.zeros((len(ee_exec), 1)); r_grip = np.zeros((len(ee_exec), 1))
            action_ee_for_save = np.concatenate(
                [l_pos, l_rot6d, l_grip, r_pos, r_rot6d, r_grip], axis=-1)  # (T, 20)
        else:
            grip = raw_actions[:, 15:16] if raw_actions.shape[1] == 16 else raw_actions[:, -1:]
            pos = ee_exec[:, :3]; quat = ee_exec[:, 3:7]
            rot6d = quat_to_6d_numpy(quat)
            action_ee_for_save = np.concatenate([pos, rot6d, grip], axis=-1)  # (T, 10)

    stride = args.umi_subsample if is_umi else args.frank_subsample
    thresh = 0.015 if is_umi else 0.005 
    
    T = min(len(obj_traj), len(ee_idm_traj), len(real_action_for_save), len(action_ee_for_save), len(joint_traj), len(target_traj))
    if args.adaptive_subsample:
        # Without an object pose, detect motion onset from the EE trajectory
        if use_obj_for_adapt: indices, _ = get_adaptive_indices(obj_traj[:T], stride, move_thresh=thresh)
        else: indices, _ = get_adaptive_indices(ee_idm_traj[:T], stride, move_thresh=thresh)
    else: indices = np.arange(0, T, stride)
    
    if len(indices) < 2: return None

    mask_dirs_dict = {v: Path(file_path).parent / f"{v}_frames_mask" for v in args.camera_views}
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    image_mode = getattr(args, 'image_mode', 'mask')
    imgs = []
    for idx in indices:
        imgs.append(load_multi_view_mask(mask_dirs_dict, idx + trim_start, transform, args.camera_views, image_mode=image_mode))
    img_mask_sub = torch.stack(imgs)

    return {
        'obj_raw': obj_traj[indices], 'ee_raw': ee_idm_traj[indices], 'joint_raw': joint_traj[indices],
        'target_raw': target_traj[indices], 'action_raw': real_action_for_save[indices],
        'action_ee_raw': action_ee_for_save[indices], 'frame_indices': indices.astype(np.int64),
        'img_mask_raw': img_mask_sub,
    }

def _pad_chunk(data_array, t, K):
    T_len = len(data_array)
    end_idx = min(t + K, T_len)
    actual_len = end_idx - t
    if actual_len == K: return data_array[t : end_idx]
    if isinstance(data_array, torch.Tensor):
        chunk = torch.zeros((K, *data_array.shape[1:]), dtype=data_array.dtype, device=data_array.device)
    else: chunk = np.zeros((K, *data_array.shape[1:]), dtype=data_array.dtype)
    if actual_len > 0: chunk[:actual_len] = data_array[t : end_idx]
    last_frame = data_array[min(end_idx - 1, T_len - 1)]
    for p in range(actual_len, K): chunk[p] = last_frame
    return chunk

def process_roots(roots, model, normalizer, args, data_type="frank"):
    device = torch.device(args.device)
    total_success = 0
    print(f"\n[{data_type.upper()}] Processing roots: {roots}")
    K = getattr(args, 'chunk_size', 1)
    is_umi = (data_type == "umi")

    for root_dir in roots:
        root_path = Path(root_dir)
        if not root_path.exists(): continue
        all_files = sorted(list(root_path.rglob("*.npz")))
        if args.tasks != 'all':
            task_list = [f"task_{int(t):02d}" for t in args.tasks.split()]
            target_files = [f for f in all_files if any(t in f.parts for t in task_list)]
        else: target_files = all_files
            
        print(f"  -> Found {len(target_files)} files in {root_path.name}")
        for file_path in tqdm(target_files, desc=f"Labeling {data_type}"):
            batch_data = load_and_prep_file(file_path, args, normalizer, data_type=data_type)
            if batch_data is None: continue
            
            obj = batch_data['obj_raw']; ee = batch_data['ee_raw']
            valid_len = len(obj) - 1
            if valid_len <= 0: continue

            ee_t_list, ee_tp1_list, img_t_list, img_tp1_list = [], [], [], []
            for t in range(valid_len):
                if K > 1:
                    ee_t_list.append(_pad_chunk(ee, t, K)); ee_tp1_list.append(_pad_chunk(ee, t+1, K))
                    img_t_list.append(_pad_chunk(batch_data['img_mask_raw'], t, K))
                    img_tp1_list.append(_pad_chunk(batch_data['img_mask_raw'], t+1, K))
                else:
                    ee_t_list.append(ee[t]); ee_tp1_list.append(ee[t+1])
                    img_t_list.append(batch_data['img_mask_raw'][t]); img_tp1_list.append(batch_data['img_mask_raw'][t+1])

            ee_t = torch.from_numpy(np.array(ee_t_list)).float().to(device)
            ee_tp1 = torch.from_numpy(np.array(ee_tp1_list)).float().to(device)
            ee_t_norm = normalizer.normalize('ee_idm', ee_t); ee_tp1_norm = normalizer.normalize('ee_idm', ee_tp1)

            img_t_tensor = torch.stack(img_t_list).to(device)
            img_tp1_tensor = torch.stack(img_tp1_list).to(device)

            B = img_t_tensor.shape[0]

            with torch.no_grad():
                obj_t_feat = model.encode_vision(img_t_tensor); obj_tp1_feat = model.encode_vision(img_tp1_tensor)

                obj_t_idm = obj_t_feat[:, 0] if K > 1 else obj_t_feat
                obj_tp1_idm = obj_tp1_feat[:, -1] if K > 1 else obj_tp1_feat

                if args.idm_obj_only: idm_in = torch.cat([obj_t_idm, obj_tp1_idm], dim=-1)
                elif args.use_idm_ee:
                    ee_t_9d = model.process_pose_input(ee_t_norm); ee_t_idm = ee_t_9d[:, 0] if K > 1 else ee_t_9d
                    if getattr(model, "use_idm_ee_part", False):
                        idm_in = torch.cat([obj_t_idm, ee_t_idm, obj_tp1_idm], dim=-1)
                    else:
                        ee_tp1_9d = model.process_pose_input(ee_tp1_norm); ee_tp1_idm = ee_tp1_9d[:, -1] if K > 1 else ee_tp1_9d
                        idm_in = torch.cat([obj_t_idm, ee_t_idm, obj_tp1_idm, ee_tp1_idm], dim=-1)
                else: idm_in = torch.cat([obj_t_idm, obj_tp1_idm], dim=-1)
                
                mu_flat, log_std_flat = model.idm(idm_in)
                mu = mu_flat.view(B, K, -1) if K > 1 else mu_flat
                log_std = log_std_flat.view(B, K, -1) if K > 1 else log_std_flat
                
                z_latent = mu.cpu().numpy(); z_log_std = log_std.cpu().numpy()
            
            # Mirror the source layout under out_root/<data_type>/
            rel_path = file_path.relative_to(root_path)
            save_dir = Path(args.out_root) / data_type / root_path.name / rel_path.parent
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "policy_data.npz"
            
            np.savez(
                save_path,
                obs_obj=batch_data['obj_raw'][:valid_len], obs_ee=batch_data['ee_raw'][:valid_len],
                obs_joint=batch_data['joint_raw'][:valid_len], obs_target=batch_data['target_raw'][:valid_len],
                action_latent_mean=z_latent, action_latent_logstd=z_log_std,
                action_real=batch_data['action_raw'][:valid_len], action_ee_real=batch_data['action_ee_raw'][:valid_len],
                frame_indices=batch_data['frame_indices'][:valid_len],
            )
            total_success += 1
            
    return total_success

def main(args):
    device = torch.device(args.device)
    print(f"Loading Checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        sync_keys = ['latent_dim', 'robot_state_type', 'action_type', 'use_idm_ee', 'use_idm_ee_part', 'idm_obj_only', 'camera_view', 'chunk_size', 'camera_views',
                    'raw_image', 'masked_color_image']
        for k in sync_keys:
            if k in saved_args: setattr(args, k, saved_args[k])
            else:
                if k == 'chunk_size': setattr(args, k, 1)
                elif k == 'camera_views': setattr(args, k, ['front'])
                elif k in ('raw_image', 'masked_color_image'): setattr(args, k, False)

    # Determine image_mode from synced flags
    image_mode = 'mask'
    if getattr(args, 'masked_color_image', False): image_mode = 'masked_color'
    elif getattr(args, 'raw_image', False): image_mode = 'raw'
    args.image_mode = image_mode
    print(f"[GLAM-generate] Using image_mode from checkpoint: {image_mode}")

    print(f"GLAM chunk size configured to: {getattr(args, 'chunk_size', 1)}")
    if 'stats' not in checkpoint: raise ValueError("No stats in ckpt!")
    normalizer = Normalizer(checkpoint['stats'], robot_state_type=args.robot_state_type, action_type=args.action_type)
    
    s_dim = normalizer.stats['robot_state'][0].shape[0]
    a_dim = normalizer.stats['robot_action'][0].shape[0]
    ee_internal_dim = 18 if (normalizer.stats.get('ee_idm') is not None and normalizer.stats['ee_idm'][0].shape[0] == 6) else 9
    print(f"[CONFIG] s_dim={s_dim}, a_dim={a_dim}, ee_internal_dim={ee_internal_dim}")

    model = GLAM(
        robot_state_dim=s_dim, robot_action_dim=a_dim, action_latent_dim=args.latent_dim,
        use_idm_ee=args.use_idm_ee, use_idm_ee_part=args.use_idm_ee_part, idm_obj_only=getattr(args, 'idm_obj_only', False),
        chunk_size=getattr(args, 'chunk_size', 1),
        num_cameras=len(getattr(args, 'camera_views', ['front'])),
        image_mode=image_mode,
        ee_internal_dim=ee_internal_dim,
    ).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=False); model.eval()
    
    if args.frank_roots: process_roots(args.frank_roots, model, normalizer, args, data_type="frank")
    if args.frank_real_roots: process_roots(args.frank_real_roots, model, normalizer, args, data_type="frank_real")
    if args.umi_roots: process_roots(args.umi_roots, model, normalizer, args, data_type="umi")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--frank_roots", nargs='+', help="List of Frank dataset directories")
    parser.add_argument("--frank_real_roots", nargs='+', help="List of Frank Real dataset directories")
    parser.add_argument("--umi_roots", nargs='+', help="List of UMI dataset directories")

    parser.add_argument("--tasks", type=str, default="all")
    parser.add_argument("--frank_subsample", type=int, default=1)
    parser.add_argument("--frank_real_subsample", type=int, default=1)
    parser.add_argument("--umi_subsample", type=int, default=1)
    parser.add_argument("--adaptive_subsample", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--action_type", type=str, default="joint")
    parser.add_argument("--robot_state_type", type=str, default="ee")
    parser.add_argument("--camera_views", nargs='+', default=["front"], help="List of views e.g. front overhead")
    parser.add_argument("--use_idm_ee", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=1)
    args = parser.parse_args()
    main(args)

# Example:
# python glam/generate_policy_dataset.py \
#     --umi_roots data/stack_two/umi/task_01/ --frank_roots data/stack_two/frank/task_01/ \
#     --tasks 1 --umi_subsample 1 --frank_subsample 1 \
#     --ckpt checkpoints/glam/<run_name>/best_val.pth \
#     --out_root data/stack_two/policy_dataset