import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from vision_encoder import get_vision_transform

class Normalizer:
    def __init__(self, stats=None, robot_state_type='joint', action_type='joint'):
        self.stats = stats if stats is not None else {}
        self.eps = 1e-6
        self.robot_state_type = robot_state_type
        self.action_type = action_type

    def normalize(self, key, data):
        if key not in self.stats: return data
        mean, std = self.stats[key]
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device; mean = torch.from_numpy(mean).to(device).float(); std = torch.from_numpy(std).to(device).float()
        if key == 'action' and self.action_type in ['ee', 'ee_delta']:
            if data.shape[-1] == 10:
                pos = data[..., :3]; rot6d = data[..., 3:9]; grip = data[..., 9:]
                pos_mean = mean[..., :3]; pos_std = std[..., :3]
                grip_mean = mean[..., 9:]; grip_std = std[..., 9:]
                pos_norm = (pos - pos_mean) / (pos_std + self.eps)
                grip_norm = (grip - grip_mean) / (grip_std + self.eps)
                if is_tensor: return torch.cat([pos_norm, rot6d, grip_norm], dim=-1)
                return np.concatenate([pos_norm, rot6d, grip_norm], axis=-1)
        return (data - mean) / (std + self.eps)

    def unnormalize(self, key, data):
        if key not in self.stats: return data
        mean, std = self.stats[key]
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device; mean = torch.from_numpy(mean).to(device).float(); std = torch.from_numpy(std).to(device).float()
        if key == 'action' and self.action_type in ['ee', 'ee_delta']:
            if data.shape[-1] == 10:
                pos = data[..., :3]; rot6d = data[..., 3:9]; grip = data[..., 9:]
                pos_mean = mean[..., :3]; pos_std = std[..., :3]
                grip_mean = mean[..., 9:]; grip_std = std[..., 9:]
                pos_un = pos * (pos_std + self.eps) + pos_mean
                grip_un = grip * (grip_std + self.eps) + grip_mean
                if is_tensor: return torch.cat([pos_un, rot6d, grip_un], dim=-1)
                return np.concatenate([pos_un, rot6d, grip_un], axis=-1)
        return data * (std + self.eps) + mean

class PolicyDataset(Dataset):
    def __init__(self, data_root, split="train", val_ratio=0.1, seed=42,
                 frank_max_episodes=-1, umi_max_episodes=-1, frank_real_max_episodes=-1,
                 act_chunk=1, action_type="joint", robot_state_type="joint", base_policy=False,
                 image_root="", camera_views="front", image_size=224,
                 freeze_vision_encoder=True, vision_backbone="dinov2_vits14"):
        super().__init__()
        self.data_root = Path(data_root)
        self.split = split
        self.act_chunk = int(act_chunk)
        self.action_type = action_type; self.robot_state_type = robot_state_type; self.base_policy = base_policy

        self.image_root = image_root; self.camera_views = camera_views
        self.image_size = image_size; self.freeze_vision_encoder = freeze_vision_encoder; self.vision_backbone = vision_backbone
        self.img_transform = get_vision_transform(image_size)

        all_files = sorted(list(self.data_root.rglob("policy_data.npz")))
        if len(all_files) == 0: raise ValueError(f"No policy_data.npz found in {data_root}")

        # Split files by data source
        frank_sim_files = [f for f in all_files if "frank" in f.parts and "frank_real" not in f.parts]
        frank_real_files = [f for f in all_files if "frank_real" in f.parts]
        umi_files = [f for f in all_files if "umi" in f.parts]

        if frank_max_episodes >= 0: frank_sim_files = frank_sim_files[:frank_max_episodes]
        if frank_real_max_episodes >= 0: frank_real_files = frank_real_files[:frank_real_max_episodes]
        if umi_max_episodes >= 0: umi_files = umi_files[:umi_max_episodes]

        self.files = sorted(frank_sim_files + frank_real_files + umi_files)
        print(f"[{split.upper()}] Loaded: Sim({len(frank_sim_files)}), Real({len(frank_real_files)}), UMI({len(umi_files)})")

        self.ep_obs_joint = []
        self.ep_latent = []; self.ep_action = []; self.ep_len = []
        self.ep_data_source = []  # 0=UMI, 1=Frank_Sim, 2=Frank_Real
        self.ep_frame_indices = []; self.ep_image_dirs = []; self.ep_vision_feats = []

        loaded = 0
        for f in self.files:
            try:
                data = np.load(f, allow_pickle=True)

                if "frank_real" in f.parts: data_source = 2
                elif "frank" in f.parts: data_source = 1
                elif "umi" in f.parts: data_source = 0
                else: data_source = -1

                latent = data["action_latent_mean"].astype(np.float32)
                T = len(latent)

                if "obs_joint" in data:
                    joint_data = data["obs_joint"].astype(np.float32)
                else:
                    # Infer dim from EE: if EE is 14d (dual-arm), joint should be 16d; else 8d single-arm
                    inferred_joint_dim = 16 if ("obs_ee" in data and data["obs_ee"].shape[-1] == 14) else 8
                    joint_data = np.zeros((T, inferred_joint_dim), dtype=np.float32)
                if self.action_type in ['ee', 'ee_delta']: action = data["action_ee_real"].astype(np.float32) if "action_ee_real" in data else data["action_real"].astype(np.float32)
                else: action = data["action_real"].astype(np.float32)

                if self.action_type == 'ee_delta':
                    d_pos = np.zeros_like(action[..., :3]); d_pos[:-1] = action[1:, :3] - action[:-1, :3]
                    d_rot = np.zeros_like(action[..., 3:9]); d_rot[:-1] = action[1:, 3:9] - action[:-1, 3:9]
                    grip = action[..., 9:]; action = np.concatenate([d_pos, d_rot, grip], axis=-1)

                self.ep_obs_joint.append(joint_data)
                self.ep_latent.append(latent); self.ep_action.append(action)
                self.ep_data_source.append(data_source); self.ep_len.append(T)

                self.ep_frame_indices.append(data["frame_indices"].astype(np.int64) if "frame_indices" in data else np.arange(T, dtype=np.int64))
                self.ep_image_dirs.append(str(f.relative_to(self.data_root).parent))
                feat_path = f.parent / f"vision_feat_{self.camera_views}_{self.vision_backbone}.npy"
                feat_path_legacy = f.parent / f"vision_feat_{self.camera_views}.npy"
                if self.freeze_vision_encoder:
                    if feat_path.exists(): self.ep_vision_feats.append(np.load(feat_path).astype(np.float32))
                    elif feat_path_legacy.exists() and self.vision_backbone == "dinov2_vits14": self.ep_vision_feats.append(np.load(feat_path_legacy).astype(np.float32))
                    else: self.ep_vision_feats.append(None)
                else: self.ep_vision_feats.append(None)
                loaded += 1
            except Exception as e: print(f"[PolicyDataset] Error loading {f}: {e}")

        if loaded == 0:
            self.samples = []; self.normalizer = Normalizer(robot_state_type=self.robot_state_type, action_type=self.action_type)
            return

        num_files = len(self.files)
        file_indices = np.arange(num_files)
        rng = np.random.default_rng(seed)
        rng.shuffle(file_indices)
        n_val = int(num_files * val_ratio)
        sel_eps = set(file_indices[n_val:].tolist()) if split == "train" else set(file_indices[:n_val].tolist())
        self.selected_files = [str(self.files[i]) for i in sorted(list(sel_eps))]

        self.samples = []
        for ep_id in range(num_files):
            if ep_id not in sel_eps: continue
            for t in range(self.ep_len[ep_id]): self.samples.append((ep_id, t))

        all_joint = np.concatenate(self.ep_obs_joint, axis=0)
        all_action = np.concatenate(self.ep_action, axis=0)

        # Joint/action stats come from robot data only (sim + real); UMI has no real robot actions
        frank_mask = np.concatenate([np.full((self.ep_len[i],), 1.0 if self.ep_data_source[i] in [1, 2] else 0.0, dtype=np.float32) for i in range(num_files)], axis=0)
        frank_idx = np.where(frank_mask > 0.5)[0]

        self.normalizer = Normalizer(robot_state_type=self.robot_state_type, action_type=self.action_type)
        valid_joint = all_joint if len(frank_idx) == 0 else all_joint[frank_idx]
        valid_action = all_action if len(frank_idx) == 0 else all_action[frank_idx]
        self.normalizer.stats["obs_joint"] = (np.mean(valid_joint, axis=0), np.std(valid_joint, axis=0))
        self.normalizer.stats["action"] = (np.mean(valid_action, axis=0), np.std(valid_action, axis=0))

        self.joint_dim = all_joint.shape[1]
        self.latent_dim = self.ep_latent[0].shape[-1]; self.action_dim = all_action.shape[1]

        self.vision_num_views = 2 if self.camera_views == "both" else 1

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        ep_id, t = self.samples[idx]

        joint = self.ep_obs_joint[ep_id]
        latent = self.ep_latent[ep_id]; action = self.ep_action[ep_id]
        data_source = self.ep_data_source[ep_id]

        joint_in = joint[t][None, :]

        K_pol = self.act_chunk
        a_gt = np.zeros((K_pol, action.shape[-1]), dtype=np.float32)
        end_idx = min(t + K_pol, len(action)); act_len = end_idx - t
        if act_len > 0: a_gt[:act_len] = action[t : end_idx]
        if act_len < K_pol:
            for pad_idx in range(act_len, K_pol): a_gt[pad_idx] = a_gt[max(0, act_len - 1)]
                
        z_gt = np.zeros((K_pol, latent.shape[-1]), dtype=np.float32)
        if not self.base_policy:
            K_glam = 1 if latent.ndim == 2 else latent.shape[1]
            if K_glam == 1:
                end_idx_z = min(t + K_pol, len(latent)); z_len = end_idx_z - t
                if z_len > 0: z_gt[:z_len] = latent[t : end_idx_z]
                if z_len < K_pol:
                    for pad_idx in range(z_len, K_pol): z_gt[pad_idx] = z_gt[max(0, z_len - 1)]
            else:
                if K_glam == K_pol: z_gt = latent[t]
                elif K_glam > K_pol: z_gt = latent[t][:K_pol]
                elif K_glam < K_pol:
                    extracted = 0; curr_t = t; last_valid = np.zeros(latent.shape[-1], dtype=np.float32)
                    while extracted < K_pol:
                        take = min(K_glam, K_pol - extracted)
                        if curr_t < len(latent):
                            z_gt[extracted : extracted + take] = latent[curr_t][:take]
                            last_valid = latent[curr_t][:take][-1]
                        else: z_gt[extracted : extracted + take] = last_valid
                        extracted += take; curr_t += K_glam

        joint_n = self.normalizer.normalize("obs_joint", joint_in)
        a_n = self.normalizer.normalize("action", a_gt)

        joint_feat = joint_n.reshape(-1).astype(np.float32)

        if self.ep_vision_feats[ep_id] is not None:
            image_tensor = torch.from_numpy(self.ep_vision_feats[ep_id][t]).float()
        else:
            frame_idx = int(self.ep_frame_indices[ep_id][t])
            img_dir = self.ep_image_dirs[ep_id]; view_tensors = []
            view_list = ["front", "overhead"] if self.camera_views == "both" else [self.camera_views]
            for vname in view_list:
                img_path = Path(self.image_root) / img_dir / f"{vname}_frames" / f"{frame_idx:04d}.png"
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_t = self.img_transform(img)
                except Exception as e:
                    # Log once per missing dir to avoid spamming, then crash.
                    if not hasattr(self, '_missing_warned'):
                        self._missing_warned = set()
                    if str(img_path.parent) not in self._missing_warned:
                        self._missing_warned.add(str(img_path.parent))
                        print(f"\n[FATAL] Missing image: {img_path}")
                        print(f"        image_root: {self.image_root}")
                        print(f"        Check that raw images have been transferred to this server.")
                    raise FileNotFoundError(f"Missing image: {img_path}") from e
                view_tensors.append(img_t)
            image_tensor = view_tensors[0] if len(view_tensors) == 1 else torch.cat(view_tensors, dim=0)

        return {
            "obs_joint": torch.from_numpy(joint_feat).float(),
            "action_latent": torch.from_numpy(z_gt).float(),
            "action_real": torch.from_numpy(a_n).float(),
            "data_source": torch.tensor(data_source, dtype=torch.long),
            "image": image_tensor,
        }
    
