"""Train a small action decoder on top of a frozen pretrained GLAM model.

The decoder maps GLAM action latents (from the IDM or the robot encoder) to
robot actions. It is used by generate_val_predictions.py to turn latents
predicted on held-out episodes into executable actions.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from pathlib import Path

from dataset import CombinedRoboticDataset, Normalizer
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


def save_checkpoint(state, path):
    torch.save(state, path)


def encode_latent(glam, batch, mask, K, encoder_source, use_idm_ee, device):
    """Encode a masked batch into (mu, logvar) latents with the frozen GLAM model."""
    robot_s_raw = batch['robot_state'].to(device)[mask]
    robot_a_raw = batch['robot_action'].to(device)[mask]
    B_m = robot_s_raw.shape[0]

    if K > 1:
        robot_s = robot_s_raw.view(B_m * K, -1)
        robot_a = robot_a_raw.view(B_m * K, -1)
    else:
        robot_s = robot_s_raw
        robot_a = robot_a_raw

    if encoder_source == 'robot':
        mu, logvar = glam.robot_enc(torch.cat([robot_s, robot_a], dim=-1))
        return mu, logvar, B_m

    # encoder_source == 'idm'
    img_t = batch['img_mask_t'].to(device)[mask]
    img_tp1 = batch['img_mask_tp1'].to(device)[mask]
    obj_t_feat = glam.encode_vision(img_t)
    obj_tp1_feat = glam.encode_vision(img_tp1)
    obj_t_idm = obj_t_feat[:, 0] if K > 1 else obj_t_feat
    obj_tp1_idm = obj_tp1_feat[:, -1] if K > 1 else obj_tp1_feat

    ee_t_idm = None; ee_tp1_idm = None
    if use_idm_ee:
        ee_t = glam.process_pose_input(batch['ee_t'].to(device)[mask])
        ee_t_idm = ee_t[:, 0] if K > 1 else ee_t
        if not getattr(glam, "use_idm_ee_part", False):
            ee_tp1 = glam.process_pose_input(batch['ee_tp1'].to(device)[mask])
            ee_tp1_idm = ee_tp1[:, -1] if K > 1 else ee_tp1

    idm_in = glam.build_idm_input(obj_t_idm, obj_tp1_idm, ee_t_idm, ee_tp1_idm)
    mu_flat, logvar_flat = glam.idm(idm_in)
    mu = mu_flat.view(B_m * K, -1) if K > 1 else mu_flat
    logvar = logvar_flat.view(B_m * K, -1) if K > 1 else logvar_flat
    return mu, logvar, B_m


def evaluate(glam, decoder, val_loader, device, args):
    decoder.eval()
    total_loss = 0; count = 0
    K = args.chunk_size
    with torch.no_grad():
        for batch in val_loader:
            M = batch['M'].to(device)
            mask = M > 0.5
            if mask.sum() == 0: continue

            target_a_raw = batch['robot_action_target'].to(device)[mask]
            mu, _, B_m = encode_latent(glam, batch, mask, K, args.encoder_source, args.use_idm_ee, device)
            target_a = target_a_raw.view(B_m * K, -1) if K > 1 else target_a_raw

            pred_action = decoder(mu)
            loss = F.mse_loss(pred_action, target_a)

            actual_items = B_m * K if K > 1 else B_m
            total_loss += loss.item() * actual_items
            count += actual_items

    return total_loss / max(count, 1)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading GLAM from: {args.glam_ckpt}")
    checkpoint = torch.load(args.glam_ckpt, map_location=device)

    if 'args' in checkpoint:
        saved_args = checkpoint['args']
        sync_keys = [
            'latent_dim', 'robot_state_type', 'action_type',
            'forward_robot', 'use_idm_ee', 'use_idm_ee_part',
            'idm_obj_only', 'strict_robot_encoder',
            'recon_action', 'recon_action_rob', 'rob_latent_forward',
            'robot_as_posterior', 'camera_views', 'chunk_size',
            'masked_color_image', 'raw_image'
        ]
        for k in sync_keys:
            if k in saved_args: setattr(args, k, saved_args[k])
            else:
                if k in ['forward_robot', 'use_idm_ee', 'strict_robot_encoder', 'recon_action', 'masked_color_image', 'raw_image']: setattr(args, k, False)
                elif k == 'chunk_size': setattr(args, k, 1)

    image_mode = 'mask'
    if getattr(args, 'masked_color_image', False): image_mode = 'masked_color'
    elif getattr(args, 'raw_image', False): image_mode = 'raw'

    glam_run_name = Path(args.glam_ckpt).parent.name
    run_name = f"ActDec_{args.target_action_type}_from_{args.encoder_source}__for__{glam_run_name}"
    final_save_dir = Path(args.save_dir) / run_name
    final_save_dir.mkdir(parents=True, exist_ok=True)

    if args.use_wandb: wandb.init(project="glam_action_decoder", config=args, name=run_name)

    stats = checkpoint['stats']
    normalizer = Normalizer(stats, robot_state_type=args.robot_state_type)
    s_dim = stats['robot_state'][0].shape[0]
    a_dim_enc = stats['robot_action'][0].shape[0]
    # Auto-detect dual-arm from ee_idm stats: single-arm stores 3-d pos mean, dual-arm stores 6-d (l+r).
    ee_internal_dim = 18 if (stats.get('ee_idm') is not None and stats['ee_idm'][0].shape[0] == 6) else 9
    print(f"[CONFIG] s_dim={s_dim}, a_dim={a_dim_enc}, ee_internal_dim={ee_internal_dim}")

    glam = GLAM(
        robot_state_dim=s_dim, robot_action_dim=a_dim_enc, action_latent_dim=args.latent_dim,
        use_robot_forward=args.forward_robot, use_idm_ee=args.use_idm_ee,
        use_idm_ee_part=args.use_idm_ee_part, idm_obj_only=getattr(args, 'idm_obj_only', False),
        strict_robot_encoder=args.strict_robot_encoder, use_action_recon=args.recon_action,
        use_rob_action_recon=getattr(args, 'recon_action_rob', False),
        rob_latent_forward=getattr(args, 'rob_latent_forward', False), chunk_size=args.chunk_size,
        num_cameras=len(args.camera_views), image_mode=image_mode,
        ee_internal_dim=ee_internal_dim
    ).to(device)

    glam.load_state_dict(checkpoint.get('model_state', checkpoint))
    glam.eval()
    for param in glam.parameters(): param.requires_grad = False

    tasks = 'all' if args.frank_tasks == 'all' else [int(x) for x in args.frank_tasks.split()]
    umi_tasks = 'all' if args.umi_task_ids == 'all' else [int(x) for x in args.umi_task_ids.split()]
    frank_real_tasks = 'all' if args.frank_real_tasks == 'all' else [int(x) for x in args.frank_real_tasks.split()]

    train_ds = CombinedRoboticDataset(
        umi_roots=args.umi_roots, frank_roots=args.frank_roots, frank_real_roots=args.frank_real_roots,
        umi_subsample=args.umi_subsample, frank_subsample=args.frank_subsample,
        split='train', val_ratio=args.val_ratio, action_type=args.action_type,
        target_action_type=args.target_action_type, robot_state_type=args.robot_state_type,
        task_ids=tasks, umi_task_ids=umi_tasks, frank_real_task_ids=frank_real_tasks,
        adaptive_subsample=args.adaptive_subsample,
        frank_max_episodes=args.frank_max_episodes, umi_max_episodes=args.umi_max_episodes,
        frank_real_max_episodes=args.frank_real_max_episodes,
        normalizer=normalizer, chunk_size=args.chunk_size, camera_views=args.camera_views,
        image_mode=image_mode
    )

    target_action_dim = train_ds.normalizer.stats['robot_action_target'][0].shape[0]
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=16, pin_memory=True)

    val_ds = CombinedRoboticDataset(
        umi_roots=args.umi_roots, frank_roots=args.frank_roots, frank_real_roots=args.frank_real_roots,
        umi_subsample=args.umi_subsample, frank_subsample=args.frank_subsample,
        split='val', val_ratio=args.val_ratio, action_type=args.action_type,
        target_action_type=args.target_action_type, robot_state_type=args.robot_state_type,
        task_ids=tasks, umi_task_ids=umi_tasks, frank_real_task_ids=frank_real_tasks,
        adaptive_subsample=args.adaptive_subsample, frank_max_episodes=args.frank_max_episodes,
        frank_real_max_episodes=args.frank_real_max_episodes,
        umi_max_episodes=args.umi_max_episodes, normalizer=train_ds.normalizer, chunk_size=args.chunk_size,
        camera_views=args.camera_views, image_mode=image_mode
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    decoder = ActionDecoder(args.latent_dim, target_action_dim, hidden_dim=256, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    K = args.chunk_size
    for epoch in range(args.epochs):
        decoder.train()
        total_loss = 0; steps = 0
        for batch in train_loader:
            M = batch['M'].to(device)
            mask = M > 0.5
            if mask.sum() == 0: continue

            target_a_raw = batch['robot_action_target'].to(device)[mask]

            with torch.no_grad():
                mu, logvar, B_m = encode_latent(glam, batch, mask, K, args.encoder_source, args.use_idm_ee, device)
                z, _, _ = glam.reparameterize(mu, logvar)

            target_a = target_a_raw.view(B_m * K, -1) if K > 1 else target_a_raw

            pred = decoder(z)
            loss = F.mse_loss(pred, target_a)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); steps += 1

        avg_loss = total_loss / max(1, steps)
        val_loss = evaluate(glam, decoder, val_loader, device, args)
        print(f"Epoch {epoch+1} | Train: {avg_loss:.5f} | Val: {val_loss:.5f}")
        if args.use_wandb: wandb.log({"train_loss": avg_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({'decoder_state': decoder.state_dict(), 'best_val_loss': best_val_loss, 'stats': train_ds.normalizer.stats, 'args': vars(args)}, final_save_dir / "best_decoder.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glam_ckpt", type=str, required=True)
    parser.add_argument("--umi_roots", nargs='+', required=True)
    parser.add_argument("--frank_roots", nargs='+', required=True)
    parser.add_argument("--frank_real_roots", nargs='+', default=None)
    parser.add_argument("--save_dir", type=str, default="action_decoder_ckpts")

    parser.add_argument("--target_action_type", type=str, required=True, choices=["joint", "ee", "ee_delta"])
    parser.add_argument("--encoder_source", type=str, default="robot", choices=["robot", "idm"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model configuration is synced from the GLAM checkpoint at startup;
    # these flags only act as fallbacks for checkpoints without saved args.
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--robot_state_type", type=str, default="joint")
    parser.add_argument("--action_type", type=str, default="joint")
    parser.add_argument("--forward_robot", action="store_true")
    parser.add_argument("--use_idm_ee", action="store_true")
    parser.add_argument("--use_idm_ee_part", action="store_true")
    parser.add_argument("--idm_obj_only", action="store_true")
    parser.add_argument("--recon_action_rob", action="store_true")
    parser.add_argument("--rob_latent_forward", action="store_true")
    parser.add_argument("--robot_as_posterior", action="store_true")
    parser.add_argument("--strict_robot_encoder", action="store_true")
    parser.add_argument("--recon_action", action="store_true")
    parser.add_argument("--camera_views", nargs='+', default=["overhead"], help="List of views e.g. front overhead")
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--masked_color_image", action="store_true")
    parser.add_argument("--raw_image", action="store_true")

    parser.add_argument("--umi_subsample", type=int, default=1)
    parser.add_argument("--frank_subsample", type=int, default=1)
    parser.add_argument("--frank_real_subsample", type=int, default=1)
    parser.add_argument("--frank_tasks", type=str, default="all")
    parser.add_argument("--frank_real_tasks", type=str, default="all")
    parser.add_argument("--umi_task_ids", type=str, default="all")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--adaptive_subsample", action="store_true")
    parser.add_argument("--frank_max_episodes", type=int, default=-1)
    parser.add_argument("--frank_real_max_episodes", type=int, default=-1)
    parser.add_argument("--umi_max_episodes", type=int, default=-1)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()
    train(args)
