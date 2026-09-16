import torch
import torch.nn as nn
from vision_encoder import VisionEncoder


class PolicyNetwork(nn.Module):
    """
    Generic MLP policy backbone.
    Input: vision features + joint state (already flattened if chunked obs).
    Output: flattened vector, caller decides reshape.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=4, use_mip=False):
        super().__init__()
        self.use_mip = use_mip

        if self.use_mip:
            input_dim += output_dim + 1

        layers = []
        curr_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.Mish())
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs_vis, obs_joint, in_act=None, t=None):
        inputs = [obs_vis, obs_joint]

        if self.use_mip:
            assert in_act is not None and t is not None, "MIP requires 'in_act' and 't' inputs."
            inputs.append(in_act)
            inputs.append(t)

        x = torch.cat(inputs, dim=-1)
        return self.net(x)


class ActionDecoder(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256, num_layers=4):
        super().__init__()
        input_dim = latent_dim
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class GuidedPolicy(nn.Module):
    """
    Predict latent(s) then decode to action(s).
    Supports chunked prediction: act_chunk = K.
    """
    def __init__(
        self,
        input_dim,
        latent_dim,
        action_dim,
        act_chunk: int = 1,
        decoder_path=None,
        train_decoder=False,
        hidden_dim=256,
        policy_layers=5,
        decoder_layers=4,
        use_mip=False,
    ):
        super().__init__()
        assert act_chunk >= 1
        self.act_chunk = int(act_chunk)
        self.latent_dim = int(latent_dim)
        self.action_dim = int(action_dim)
        self.train_decoder = bool(train_decoder)
        self.use_mip = use_mip

        self.policy = PolicyNetwork(
            input_dim=input_dim,
            output_dim=self.act_chunk * self.latent_dim,
            hidden_dim=hidden_dim,
            num_layers=policy_layers,
            use_mip=self.use_mip
        )
        self.decoder = ActionDecoder(
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dim=hidden_dim,
            num_layers=decoder_layers,
        )

        if decoder_path:
            print(f"Loading Decoder from: {decoder_path}")
            checkpoint = torch.load(decoder_path, map_location="cpu")
            if "model_state" in checkpoint:
                full_state = checkpoint["model_state"]
            elif "decoder_state" in checkpoint:
                full_state = checkpoint["decoder_state"]
            else:
                full_state = checkpoint

            state_to_load = {}
            idm_prefix = "idm_action_decoder."
            rob_prefix = "rob_action_decoder."

            if any(k.startswith(idm_prefix) for k in full_state.keys()):
                for k, v in full_state.items():
                    if k.startswith(idm_prefix):
                        state_to_load[k[len(idm_prefix):]] = v
            elif any(k.startswith(rob_prefix) for k in full_state.keys()):
                for k, v in full_state.items():
                    if k.startswith(rob_prefix):
                        state_to_load[k[len(rob_prefix):]] = v
            else:
                state_to_load = full_state

            self.decoder.load_state_dict(state_to_load)
        else:
            print("No Decoder Checkpoint provided. Initializing from scratch.")

        if not self.train_decoder:
            print(">> Decoder is FROZEN.")
            for p in self.decoder.parameters():
                p.requires_grad = False
            self.decoder.eval()
        else:
            print(">> Decoder is TRAINABLE.")
            for p in self.decoder.parameters():
                p.requires_grad = True
            self.decoder.train()

    def forward(self, obs_vis, obs_joint, in_act=None, t=None):
        z_flat = self.policy(obs_vis, obs_joint, in_act, t)

        B = z_flat.shape[0]
        K = self.act_chunk
        z = z_flat.view(B, K, self.latent_dim)
        z2 = z.reshape(B * K, self.latent_dim)

        a2 = self.decoder(z2)
        a = a2.view(B, K, self.action_dim)
        return z, a


# ============================================================
# Vision-augmented Policy Wrapper
# ============================================================

class VisionPolicyWrapper(nn.Module):
    """
    Wraps a PolicyNetwork or GuidedPolicy with a VisionEncoder frontend.

    The wrapped `inner_model` is constructed with:
        input_dim = vision_feature_dim + obs_joint_dim
    """

    def __init__(
        self,
        inner_model: nn.Module,
        vision_feature_dim: int = 64,
        num_views: int = 1,
        freeze_backbone: bool = True,
        image_size: int = 224,
        vision_backbone: str = "dinov2_vits14",
    ):
        super().__init__()
        self.vision_encoder = VisionEncoder(
            output_dim=vision_feature_dim,
            num_views=num_views,
            freeze_backbone=freeze_backbone,
            image_size=image_size,
            vision_backbone=vision_backbone
        )
        self.inner_model = inner_model
        self.vision_feature_dim = vision_feature_dim

    def forward(self, images, obs_joint, in_act=None, t=None):
        """
        Parameters
        ----------
        images : Tensor (B, C, H, W)
            C=3 for single view, C=6 for both views.
        obs_joint : Tensor (B, joint_dim)
        in_act, t : for MIP
        """
        vis_feat = self.vision_encoder(images)   # (B, vision_feature_dim)

        return self.inner_model(vis_feat, obs_joint, in_act=in_act, t=t)

    @property
    def policy(self):
        """Access inner policy (for GuidedPolicy)."""
        if hasattr(self.inner_model, 'policy'):
            return self.inner_model.policy
        return self.inner_model

    @property
    def decoder(self):
        """Access decoder (for GuidedPolicy)."""
        if hasattr(self.inner_model, 'decoder'):
            return self.inner_model.decoder
        return None

    def train(self, mode=True):
        """Override to keep backbone in eval when frozen."""
        super().train(mode)
        self.vision_encoder.train(mode)   # internally handles frozen backbone
        return self