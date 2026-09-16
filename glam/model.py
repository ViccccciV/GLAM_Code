import torch
import torch.nn as nn
import torch.nn.functional as F

# ================= Rotation Math & MLP =================
def quat_to_matrix(quaternions):
    quaternions = F.normalize(quaternions, dim=-1)
    w, x, y, z = torch.unbind(quaternions, -1)
    o = torch.stack((1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w),
                     2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w),
                     2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)), -1)
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_rotation_6d(matrix): return torch.cat([matrix[..., 0], matrix[..., 1]], dim=-1)

class ResMLPBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
    def forward(self, x): return x + self.net(self.ln(x))

class ResMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([ResMLPBlock(hidden_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.mu = nn.Linear(hidden_dim, output_dim)
        self.logvar = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers: x = layer(x)
        x = self.ln_f(x)
        return self.mu(x), self.logvar(x)

class ActionDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        layers = []
        input_dim = latent_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim)); layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, z): return self.net(z)

class DynamicsDecoder(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dim, output_dim=None, num_layers=3, dual_head=False, head2_dim=None):
        super().__init__()
        self.dual_head = dual_head
        self.input_proj = nn.Linear(state_dim + latent_dim, hidden_dim)
        self.layers = nn.ModuleList([ResMLPBlock(hidden_dim) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(hidden_dim)
        out_dim1 = output_dim if output_dim else state_dim
        self.head1 = nn.Linear(hidden_dim, out_dim1)
        if self.dual_head:
            out_dim2 = head2_dim if head2_dim else out_dim1
            self.head2 = nn.Linear(hidden_dim, out_dim2)
    def forward(self, state, z):
        x = torch.cat([state, z], dim=-1); x = self.input_proj(x)
        for layer in self.layers: x = layer(x)
        x = self.ln_f(x); out1 = self.head1(x)
        if self.dual_head: return out1, self.head2(x)
        return out1

# ================= Vision Modules =================
class VisionEncoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
    def forward(self, x):
        h = self.net(x); return self.fc_mu(h), self.fc_logvar(h)

class VisionDecoder(nn.Module):
    def __init__(self, out_channels=1, latent_dim=128):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 256 * 4 * 4), nn.ReLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # outputs logits; loss applies the sigmoid
        )
    def forward(self, z):
        h = self.fc(z); h = h.view(-1, 256, 4, 4); return self.net(h)

# ================= Main Model =================
class GLAM(nn.Module):
    def __init__(self,
                 robot_state_dim, robot_action_dim, action_latent_dim=32, hidden_dim=256,
                 use_robot_forward=False, use_idm_ee=False, use_idm_ee_part=False,
                 idm_obj_only=False, strict_robot_encoder=False, use_action_recon=False,
                 use_rob_action_recon=False, rob_latent_forward=False, chunk_size=1,
                 vision_latent_dim=128, num_cameras=1,
                 image_mode='mask', ee_internal_dim=9):
        super().__init__()
        self.vision_latent_dim = vision_latent_dim
        self.num_cameras = num_cameras
        self.image_mode = image_mode

        # Channel count: mask=1 per view, masked_color/raw=3 per view
        channels_per_view = 1 if image_mode == 'mask' else 3
        self.total_img_channels = num_cameras * channels_per_view

        self.obj_internal_dim = vision_latent_dim
        self.ee_internal_dim = ee_internal_dim  # 9 for single-arm, 18 for dual-arm

        self.use_robot_forward = use_robot_forward
        self.use_idm_ee = use_idm_ee
        self.use_idm_ee_part = use_idm_ee_part
        self.strict_robot_encoder = strict_robot_encoder
        self.use_action_recon = use_action_recon
        self.use_rob_action_recon = use_rob_action_recon
        self.rob_latent_forward = rob_latent_forward
        self.idm_obj_only = idm_obj_only
        self.chunk_size = chunk_size

        self.obj_encoder = VisionEncoder(in_channels=self.total_img_channels, latent_dim=self.vision_latent_dim)
        self.obj_decoder = VisionDecoder(out_channels=self.total_img_channels, latent_dim=self.vision_latent_dim)

        if self.idm_obj_only:
            idm_in_dim = self.obj_internal_dim * 2
        else:
            if use_idm_ee:
                if use_idm_ee_part:
                    idm_in_dim = (self.obj_internal_dim * 2) + self.ee_internal_dim
                else:
                    idm_in_dim = (self.obj_internal_dim * 2) + (self.ee_internal_dim * 2)
            else:
                idm_in_dim = self.obj_internal_dim * 2

        self.idm = ResMLPEncoder(idm_in_dim, hidden_dim, action_latent_dim * self.chunk_size, num_layers=5)
        self.robot_enc = ResMLPEncoder(robot_state_dim + robot_action_dim, hidden_dim, action_latent_dim, num_layers=5)

        if self.use_idm_ee:
            fwd_in_dim = self.obj_internal_dim + self.ee_internal_dim
            self.forward_model = DynamicsDecoder(
                state_dim=fwd_in_dim, latent_dim=action_latent_dim, hidden_dim=hidden_dim,
                output_dim=self.obj_internal_dim, num_layers=4, dual_head=True,
                head2_dim=self.ee_internal_dim)
        else:
            self.forward_model = DynamicsDecoder(
                state_dim=self.obj_internal_dim, latent_dim=action_latent_dim, hidden_dim=hidden_dim,
                output_dim=self.obj_internal_dim, num_layers=4, dual_head=False)

        if self.use_robot_forward:
            self.forward_robot = DynamicsDecoder(robot_state_dim, action_latent_dim, hidden_dim, num_layers=5)

        self.idm_action_decoder = ActionDecoder(action_latent_dim, robot_action_dim, hidden_dim, 4) if self.use_action_recon else None
        self.rob_action_decoder = ActionDecoder(action_latent_dim, robot_action_dim, hidden_dim, 4) if self.use_rob_action_recon else None

    def process_pose_input(self, pose):
        last = pose.shape[-1]
        if last == 7:
            pos = pose[..., :3]; quat = pose[..., 3:7]
            rot6d = matrix_to_rotation_6d(quat_to_matrix(quat))
            return torch.cat([pos, rot6d], dim=-1)
        elif last == 14:
            l = self.process_pose_input(pose[..., 0:7])
            r = self.process_pose_input(pose[..., 7:14])
            return torch.cat([l, r], dim=-1)
        raise ValueError(f"Unexpected pose dim {last}")

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, -10, 2)
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std, std, logvar

    def encode_vision(self, img_tensor):
        orig_shape = img_tensor.shape
        if len(orig_shape) == 5:
            B, K, C, H, W = orig_shape
            img_tensor = img_tensor.view(B * K, C, H, W)
        mu, logvar = self.obj_encoder(img_tensor)
        z, _, _ = self.reparameterize(mu, logvar)
        if len(orig_shape) == 5: z = z.view(B, K, -1)
        return z

    def decode_vision(self, z):
        orig_shape = z.shape
        if len(orig_shape) == 3:
            B, K, D = orig_shape; z = z.view(B * K, D)
        img_logits = self.obj_decoder(z)
        if len(orig_shape) == 3:
            img_logits = img_logits.view(B, K, self.total_img_channels, 64, 64)
        return img_logits

    def build_idm_input(self, obj_t, obj_tp1, ee_t=None, ee_tp1=None):
        if self.idm_obj_only: return torch.cat([obj_t, obj_tp1], dim=-1)
        if self.use_idm_ee:
            if self.use_idm_ee_part: return torch.cat([obj_t, ee_t, obj_tp1], dim=-1)
            else: return torch.cat([obj_t, ee_t, obj_tp1, ee_tp1], dim=-1)
        return torch.cat([obj_t, obj_tp1], dim=-1)

    def forward(self, img_t, img_tp1, robot_s, robot_a, M, ee_t_raw=None, ee_tp1_raw=None):
        B = img_t.shape[0]; K = self.chunk_size

        obj_t = self.encode_vision(img_t); obj_tp1 = self.encode_vision(img_tp1)

        ee_t = self.process_pose_input(ee_t_raw) if (self.use_idm_ee and ee_t_raw is not None) else None
        ee_tp1 = self.process_pose_input(ee_tp1_raw) if (self.use_idm_ee and ee_tp1_raw is not None) else None

        obj_t_idm = obj_t[:, 0] if K > 1 else obj_t
        obj_tp1_idm = obj_tp1[:, -1] if K > 1 else obj_tp1
        ee_t_idm = (ee_t[:, 0] if K > 1 else ee_t) if ee_t is not None else None
        ee_tp1_idm = (ee_tp1[:, -1] if K > 1 else ee_tp1) if ee_tp1 is not None else None

        idm_input = self.build_idm_input(obj_t_idm, obj_tp1_idm, ee_t_idm, ee_tp1_idm)
        mu_idm_flat, logvar_idm_flat = self.idm(idm_input)
        mu_idm = mu_idm_flat.view(B, K, -1) if K > 1 else mu_idm_flat
        logvar_idm = logvar_idm_flat.view(B, K, -1) if K > 1 else logvar_idm_flat
        z_idm, _, _ = self.reparameterize(mu_idm, logvar_idm)

        robot_s_flat = robot_s.view(B * K, -1) if K > 1 else robot_s
        robot_a_flat = robot_a.view(B * K, -1) if K > 1 else robot_a
        mu_rob_flat, logvar_rob_flat = self.robot_enc(torch.cat([robot_s_flat, robot_a_flat], dim=-1))
        if self.strict_robot_encoder:
            M_ex = M.repeat_interleave(K).view(-1, 1) if K > 1 else M.view(-1, 1)
            mu_rob_flat *= M_ex; logvar_rob_flat *= M_ex
        mu_rob = mu_rob_flat.view(B, K, -1) if K > 1 else mu_rob_flat
        logvar_rob = logvar_rob_flat.view(B, K, -1) if K > 1 else logvar_rob_flat
        z_rob, _, _ = self.reparameterize(mu_rob, logvar_rob)

        obj_t_flat = obj_t.view(B * K, -1) if K > 1 else obj_t
        ee_t_flat = ee_t.view(B * K, -1) if (K > 1 and ee_t is not None) else ee_t
        z_idm_flat = z_idm.view(B * K, -1) if K > 1 else z_idm
        z_rob_flat = z_rob.view(B * K, -1) if K > 1 else z_rob

        if self.use_idm_ee:
            delta_obj_idm_flat, delta_ee_idm_flat = self.forward_model(torch.cat([obj_t_flat, ee_t_flat], dim=-1), z_idm_flat)
            obj_tp1_pred_idm_flat = obj_t_flat + delta_obj_idm_flat
            ee_tp1_pred_idm_flat = ee_t_flat + delta_ee_idm_flat
        else:
            delta_obj_idm_flat = self.forward_model(obj_t_flat, z_idm_flat)
            obj_tp1_pred_idm_flat = obj_t_flat + delta_obj_idm_flat
            ee_tp1_pred_idm_flat = None

        robot_s_tp1_pred_idm_flat = robot_s_flat + self.forward_robot(robot_s_flat, z_idm_flat) if self.use_robot_forward else None

        obj_tp1_pred_rob_flat, ee_tp1_pred_rob_flat, robot_s_tp1_pred_rob_flat = None, None, None
        if self.rob_latent_forward:
            if self.use_idm_ee:
                d_obj_r, d_ee_r = self.forward_model(torch.cat([obj_t_flat, ee_t_flat], dim=-1), z_rob_flat)
                obj_tp1_pred_rob_flat = obj_t_flat + d_obj_r; ee_tp1_pred_rob_flat = ee_t_flat + d_ee_r
            else:
                obj_tp1_pred_rob_flat = obj_t_flat + self.forward_model(obj_t_flat, z_rob_flat)
            if self.use_robot_forward:
                robot_s_tp1_pred_rob_flat = robot_s_flat + self.forward_robot(robot_s_flat, z_rob_flat)

        pred_action_idm_flat = self.idm_action_decoder(z_idm_flat) if self.idm_action_decoder else None
        pred_action_rob_flat = self.rob_action_decoder(z_rob_flat) if self.rob_action_decoder else None

        def unflatten(t): return t.view(B, K, -1) if t is not None and self.chunk_size > 1 else t

        ret = {
            'obj_tp1_target': obj_tp1, 'ee_tp1_target': ee_tp1,
            'obj_tp1_pred': unflatten(obj_tp1_pred_idm_flat), 'ee_tp1_pred': unflatten(ee_tp1_pred_idm_flat),
            'robot_s_tp1_pred': unflatten(robot_s_tp1_pred_idm_flat), 'pred_action_recon': unflatten(pred_action_idm_flat),
            'obj_tp1_pred_rob': unflatten(obj_tp1_pred_rob_flat), 'ee_tp1_pred_rob': unflatten(ee_tp1_pred_rob_flat),
            'robot_s_tp1_pred_rob': unflatten(robot_s_tp1_pred_rob_flat), 'pred_action_rob': unflatten(pred_action_rob_flat),
            'mu_idm': mu_idm, 'logvar_idm': logvar_idm, 'mu_rob': mu_rob, 'logvar_rob': logvar_rob,
        }
        ret['img_mask_t_recon_logits'] = self.decode_vision(obj_t)
        ret['img_mask_tp1_target_recon_logits'] = self.decode_vision(obj_tp1)
        ret['img_mask_tp1_recon_logits'] = self.decode_vision(unflatten(obj_tp1_pred_idm_flat))
        if self.rob_latent_forward:
            ret['img_mask_tp1_rob_recon_logits'] = self.decode_vision(unflatten(obj_tp1_pred_rob_flat))
        return ret