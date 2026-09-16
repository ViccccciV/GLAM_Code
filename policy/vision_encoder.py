"""
Vision Encoder for Policy Learning.

Uses DINOv2 ViT-S/14 (21M params, output 384-dim) as the backbone.
Supports single-view or dual-view (front + overhead) image inputs.
Can be frozen or finetuned.

Download:
    The model is auto-downloaded via torch.hub on first use.
    To pre-download:
        import torch
        torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
"""

import torch
import torch.nn as nn
import torchvision.transforms as T


# Standard DINOv2 preprocessing (ImageNet stats)
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD  = (0.229, 0.224, 0.225)

def get_vision_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])


class VisionEncoder(nn.Module):
    """
    DINOv2-Small backbone + trainable projection MLP.

    Parameters
    ----------
    output_dim : int
        Final feature dimension fed into the policy MLP.
    num_views : int
        1 = single camera, 2 = front + overhead (features concatenated before projection).
    freeze_backbone : bool
        If True, backbone weights are frozen; only the projection MLP trains.
    image_size : int
        DINOv2 supports multiples of 14.  224 is the standard choice.
    """

    def __init__(
        self,
        output_dim: int = 64,
        num_views: int = 1,
        freeze_backbone: bool = True,
        image_size: int = 224,
        vision_backbone: str = 'dinov2_vits14'
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_views = num_views
        self.freeze_backbone = freeze_backbone
        self.image_size = image_size
        self.vision_backbone = vision_backbone

        DINOV2_DIMS = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
        }

        # ---------- backbone ----------
        if self.vision_backbone not in DINOV2_DIMS:
            raise ValueError(f"Unsupported DINOv2 backbone: {self.vision_backbone}")

        self.backbone = torch.hub.load(
            'facebookresearch/dinov2', self.vision_backbone,
            pretrained=True,
        )
        self.backbone_dim = DINOV2_DIMS[self.vision_backbone]

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
            print(f"[VisionEncoder] Backbone FROZEN ({self.backbone_dim}-dim {self.vision_backbone}).")
        else:
            print(f"[VisionEncoder] Backbone TRAINABLE ({self.backbone_dim}-dim {self.vision_backbone}).")

        # ---------- projection MLP ----------
        proj_input_dim = self.backbone_dim * self.num_views
        self.projection = nn.Sequential(
            nn.Linear(proj_input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )
        print(f"[VisionEncoder] Projection: {proj_input_dim} -> {output_dim}  "
              f"(views={num_views}, img={image_size})")

    # ------------------------------------------------------------------
    def forward(self, images):
        """
        Parameters
        ----------
        images : Tensor
            If num_views == 1:  shape (B, 3, H, W)
            If num_views == 2:  shape (B, 6, H, W)  — front & overhead stacked on channel dim.

        Returns
        -------
        features : Tensor  (B, output_dim)
        """
        if self.freeze_backbone:
            self.backbone.eval()       # enforce eval for BN / dropout inside ViT

        if self.num_views == 1:
            feat = self._extract(images)                          # (B, 384)
        else:
            img1 = images[:, :3]       # front
            img2 = images[:, 3:]       # overhead
            f1 = self._extract(img1)   # (B, 384)
            f2 = self._extract(img2)   # (B, 384)
            feat = torch.cat([f1, f2], dim=-1)                    # (B, 768)

        return self.projection(feat)                              # (B, output_dim)

    def _extract(self, x):
        """Run backbone; return CLS token."""
        if self.freeze_backbone:
            with torch.no_grad():
                out = self.backbone(x)        # (B, 384) — dinov2 returns CLS by default
        else:
            out = self.backbone(x)
        return out

    # convenience ----------------------------------------------------------
    def train(self, mode=True):
        """Override to keep backbone in eval when frozen."""
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self
    
    def forward_from_features(self, backbone_feats):
        """Skip backbone, only run projection MLP on prefetched features."""
        return self.projection(backbone_feats)