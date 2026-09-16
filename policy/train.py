import argparse
import os
import re
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from dataset import PolicyDataset
from model import PolicyNetwork, GuidedPolicy, VisionPolicyWrapper


def set_seed(seed: int):
    """Make training reproducible across Python / NumPy / PyTorch.

    Note: full bit-level determinism on CUDA also requires
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    which slow things down and can break some ops. The defaults below are
    enough to make different seeds produce meaningfully different runs and
    the same seed produce ~the same run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_run_name(data_root, args):
    path_str = str(data_root)
    match = re.search(r"((?:F|FR)\d+_U\d+)", path_str)
    source_config = f"glam_{match.group(1)}" if match else "glam_CustomData"

    if args.frank_real_max_episodes > 0: source_config += f"_FR{args.frank_real_max_episodes}"
    if args.frank_sim_for_decoder: source_config += "_SimForDec"

    mode = "BasePolicy_MLP" if args.base_policy else f"LatentPolicy_{'TrainDec' if args.train_decoder else 'FrozenDec'}"
    if args.use_mip: mode += "_PolMIP"

    bb_short = args.vision_backbone.split('_')[-1]
    mode += f"_Vis{args.camera_views.capitalize()}_{bb_short}"
    mode += "_Frozen" if args.freeze_vision_encoder else "_FT"

    chunk_str = f"_ActC{args.act_chunk}"

    return f"{source_config}_{mode}{chunk_str}_bs{args.batch_size}_Lz{args.lambda_latent}_La{args.lambda_action}_seed{args.seed}_{args.run_note}"

def save_ckpt(state, save_dir, name): torch.save(state, os.path.join(save_dir, name))

def save_dataset_indices(dataset, save_dir, split_name):
    if hasattr(dataset, "selected_files"):
        save_path = os.path.join(save_dir, f"{split_name}_indices.txt")
        with open(save_path, "w") as f:
            for item in dataset.selected_files: f.write(f"{item}\n")

def train(args):
    set_seed(args.seed)
    print(f"[Seed] Using seed = {args.seed}")

    device = torch.device(args.device)
    if (not args.base_policy) and (not args.decoder_ckpt) and (not args.train_decoder):
        raise ValueError("For Latent Policy: Must provide --decoder_ckpt OR --train_decoder.")

    if args.base_policy:
        print("[Info] Base Policy: Forcing UMI episodes to 0.")
        args.umi_max_episodes = 0

    run_name = get_run_name(args.data_root, args)
    save_dir = Path(args.save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Start Training: {run_name}\nSaving to: {save_dir}")

    train_ds = PolicyDataset(
        args.data_root, split="train", frank_max_episodes=args.frank_max_episodes,
        umi_max_episodes=args.umi_max_episodes, frank_real_max_episodes=args.frank_real_max_episodes,
        act_chunk=args.act_chunk,
        action_type=args.action_type, robot_state_type=args.robot_state_type, base_policy=args.base_policy,
        image_root=args.image_root, camera_views=args.camera_views,
        image_size=args.image_size, freeze_vision_encoder=args.freeze_vision_encoder, vision_backbone=args.vision_backbone,
    )

    val_ds = PolicyDataset(
        args.data_root, split="val", frank_max_episodes=args.frank_max_episodes,
        umi_max_episodes=args.umi_max_episodes, frank_real_max_episodes=args.frank_real_max_episodes,
        act_chunk=args.act_chunk, action_type=args.action_type,
        robot_state_type=args.robot_state_type, base_policy=args.base_policy,
        image_root=args.image_root, camera_views=args.camera_views,
        image_size=args.image_size, freeze_vision_encoder=args.freeze_vision_encoder, vision_backbone=args.vision_backbone,
    )
    val_ds.normalizer = train_ds.normalizer

    save_dataset_indices(train_ds, save_dir, "train")
    save_dataset_indices(val_ds, save_dir, "val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

    input_dim = args.vision_feature_dim + train_ds.joint_dim

    stats = train_ds.normalizer.stats
    act_dim = train_ds.action_dim
    lat_dim = train_ds.latent_dim
    K = args.act_chunk

    if args.base_policy: inner_model = PolicyNetwork(input_dim, act_dim * K, hidden_dim=256, num_layers=10, use_mip=args.use_mip)
    else:
        inner_model = GuidedPolicy(
            input_dim=input_dim, latent_dim=lat_dim, action_dim=act_dim, act_chunk=K,
            decoder_path=args.decoder_ckpt, train_decoder=args.train_decoder, hidden_dim=256, policy_layers=10,
            use_mip=args.use_mip,
        )

    num_views = 2 if args.camera_views == "both" else 1
    model = VisionPolicyWrapper(
        inner_model=inner_model, vision_feature_dim=args.vision_feature_dim, num_views=num_views,
        freeze_backbone=args.freeze_vision_encoder, image_size=args.image_size, vision_backbone=args.vision_backbone,
    ).to(device)

    _vision_enc = model.vision_encoder
    if args.base_policy:
        _policy_net = model.inner_model
        _decoder_net = None
    else:
        _policy_net = model.inner_model.policy
        _decoder_net = model.inner_model.decoder

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    start_epoch = 0
    if args.resume:
        print(f"\n[INFO] Resuming training from checkpoint: {args.resume}")
        if not os.path.exists(args.resume): raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        inner = model.inner_model
        if args.base_policy: inner.load_state_dict(ckpt["policy_state"])
        else:
            inner.policy.load_state_dict(ckpt["policy_state"])
            if ckpt.get("decoder_state") is not None: inner.decoder.load_state_dict(ckpt["decoder_state"])

        if "vision_encoder_state" in ckpt:
            model.vision_encoder.load_state_dict(ckpt["vision_encoder_state"]); print("[INFO] Vision Encoder state loaded.")


        if "optimizer_state" in ckpt: optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"[INFO] Fast-forwarding to epoch {start_epoch} / {args.epochs}\n")

    if args.use_wandb: wandb.init(project="glam_policy_learning", name=run_name, config=vars(args))

    best_val_loss = float("inf")
    mse = nn.MSELoss(reduction="none")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if (not args.base_policy):
            if not args.train_decoder: model.inner_model.decoder.eval()

        ep_loss = {"total": 0.0, "z": 0.0, "a": 0.0}
        n_train_batches = 0

        for batch in train_loader:
            B = batch["obs_joint"].shape[0]
            obs_joint = batch["obs_joint"].to(device)

            images = batch["image"].to(device)
            if images.dim() == 2: vis_feat = _vision_enc.forward_from_features(images)
            else: vis_feat = _vision_enc(images)

            z_gt = batch["action_latent"].to(device); a_gt = batch["action_real"].to(device)

            # mask_action selects which samples supervise the action decoder / base policy
            data_source = batch["data_source"].to(device).view(-1)
            if args.frank_real_max_episodes <= 0:
                mask_action = (data_source == 1).float()  # no real data: supervise actions with sim only
            else:
                if args.frank_sim_for_decoder:
                    mask_action = ((data_source == 1) | (data_source == 2)).float()  # sim and real
                else:
                    mask_action = (data_source == 2).float()  # real only

            optimizer.zero_grad()

            if args.base_policy:
                a_gt_flat = a_gt.view(B, -1)
                if args.use_mip:
                    zeros = torch.zeros_like(a_gt_flat); t0 = torch.zeros((B, 1), device=device)
                    pred_1_flat = _policy_net(vis_feat, obs_joint, in_act=zeros, t=t0)

                    noisy_a = 0.9 * a_gt_flat + 0.1 * torch.randn_like(a_gt_flat); t1 = torch.ones((B, 1), device=device)
                    pred_2_flat = _policy_net(vis_feat, obs_joint, in_act=noisy_a, t=t1)

                    pred_1 = pred_1_flat.view(B, K, act_dim); pred_2 = pred_2_flat.view(B, K, act_dim)
                    loss_vec = (mse(pred_1, a_gt) + mse(pred_2, a_gt)).mean(dim=(1, 2)) / 2.0
                else:
                    pred_flat = _policy_net(vis_feat, obs_joint)
                    loss_vec = mse(pred_flat.view(B, K, act_dim), a_gt).mean(dim=(1, 2))
                    
                loss = (loss_vec * mask_action).sum() / (mask_action.sum() + 1e-9)
                loss.backward(); optimizer.step()
                ep_loss["total"] += float(loss.item()); ep_loss["a"] += float(loss.item())
                
            else:
                z_gt_flat = z_gt.view(B, -1)

                if args.use_mip:
                    zeros_z = torch.zeros_like(z_gt_flat); t0_z = torch.zeros((B, 1), device=device)
                    z_pred_1_flat = _policy_net(vis_feat, obs_joint, in_act=zeros_z, t=t0_z)

                    noisy_z = 0.9 * z_gt_flat + 0.1 * torch.randn_like(z_gt_flat); t1_z = torch.ones((B, 1), device=device)
                    z_pred_2_flat = _policy_net(vis_feat, obs_joint, in_act=noisy_z, t=t1_z)

                    # Latent loss is unmasked: every data source supervises the latent prediction
                    loss_z = (mse(z_pred_1_flat, z_gt_flat) + mse(z_pred_2_flat, z_gt_flat)).mean() / 2.0
                    z_for_decoding = z_pred_2_flat.view(B * K, lat_dim)
                else:
                    z_pred_flat = _policy_net(vis_feat, obs_joint)
                    loss_z = mse(z_pred_flat, z_gt_flat).mean()
                    z_for_decoding = z_pred_flat.view(B * K, lat_dim)

                a_pred_step = _decoder_net(z_for_decoding)
                loss_a_vec = mse(a_pred_step.view(B, K, act_dim), a_gt).mean(dim=(1, 2))

                loss_a = (loss_a_vec * mask_action).sum() / (mask_action.sum() + 1e-9)
                total_loss = args.lambda_latent * loss_z + args.lambda_action * loss_a
                
                total_loss.backward(); optimizer.step()
                ep_loss["total"] += float(total_loss.item()); ep_loss["z"] += float(loss_z.item()); ep_loss["a"] += float(loss_a.item())

            n_train_batches += 1

        avg_train = {k: v / max(1, n_train_batches) for k, v in ep_loss.items()}

        # ---------------- Validation ----------------
        model.eval()
        val_loss_a_sum, val_loss_z_sum, val_total_sum, n_val_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                B = batch["obs_joint"].shape[0]
                obs_joint = batch["obs_joint"].to(device)

                images = batch["image"].to(device)
                if images.dim() == 2: vis_feat = _vision_enc.forward_from_features(images)
                else: vis_feat = _vision_enc(images)

                z_gt = batch["action_latent"].to(device); a_gt = batch["action_real"].to(device)

                data_source_val = batch["data_source"].to(device).view(-1)
                if args.frank_real_max_episodes <= 0:
                    mask_action_val = (data_source_val == 1).float()
                else:
                    if args.frank_sim_for_decoder: mask_action_val = ((data_source_val == 1) | (data_source_val == 2)).float()
                    else: mask_action_val = (data_source_val == 2).float()

                if args.base_policy:
                    if args.use_mip:
                        zeros = torch.zeros((B, K * act_dim), device=device); t0 = torch.zeros((B, 1), device=device)
                        pred_1_flat = _policy_net(vis_feat, obs_joint, in_act=zeros, t=t0)
                        in_act_2 = 0.9 * pred_1_flat; t1 = torch.ones((B, 1), device=device)
                        pred_a = _policy_net(vis_feat, obs_joint, in_act=in_act_2, t=t1).view(B, K, act_dim)
                    else:
                        pred_a = _policy_net(vis_feat, obs_joint).view(B, K, act_dim)
                        
                    l_full = mse(pred_a, a_gt).mean(dim=(1, 2))
                    l_a = (l_full * mask_action_val).sum() / (mask_action_val.sum() + 1e-9)

                    val_loss_a_sum += float(l_a.item()); val_total_sum += float(l_a.item())
                else:
                    if args.use_mip:
                        zeros_z = torch.zeros((B, K * lat_dim), device=device); t0_z = torch.zeros((B, 1), device=device)
                        z_pred_1_flat = _policy_net(vis_feat, obs_joint, in_act=zeros_z, t=t0_z)
                        in_act_z_2 = 0.9 * z_pred_1_flat; t1_z = torch.ones((B, 1), device=device)
                        z_final_flat = _policy_net(vis_feat, obs_joint, in_act=in_act_z_2, t=t1_z)
                    else: z_final_flat = _policy_net(vis_feat, obs_joint)
                        
                    z_final_step = z_final_flat.view(B * K, lat_dim)
                    a_final_step = _decoder_net(z_final_step)

                    z_pred = z_final_flat.view(B, K, lat_dim); a_pred = a_final_step.view(B, K, act_dim)
                        
                    l_z = mse(z_pred, z_gt).mean()
                    l_a_full = mse(a_pred, a_gt).mean(dim=(1, 2))
                    l_a = (l_a_full * mask_action_val).sum() / (mask_action_val.sum() + 1e-9)

                    total = args.lambda_latent * l_z + args.lambda_action * l_a
                    val_loss_z_sum += float(l_z.item()); val_loss_a_sum += float(l_a.item()); val_total_sum += float(total.item())
                n_val_batches += 1

        avg_val_a = val_loss_a_sum / max(1, n_val_batches)
        avg_val_total = val_total_sum / max(1, n_val_batches)


        print(f"Epoch {epoch+1:03d} | Train: {avg_train['total']:.4f} | Val Act: {avg_val_a:.4f} | Val Total: {avg_val_total:.4f}")

        if args.use_wandb:
            wandb.log({
                "train/loss": avg_train["total"], "train/loss_z": avg_train["z"], "train/loss_a": avg_train["a"],
                "val/loss_total": avg_val_total, "val/loss_a": avg_val_a,
                "val/loss_z": (val_loss_z_sum / max(1, n_val_batches)) if (not args.base_policy) else 0.0,
                "epoch": epoch,
            })

        inner = model.inner_model
        policy_state = inner.state_dict() if args.base_policy else inner.policy.state_dict()
        decoder_state = None if args.base_policy else inner.decoder.state_dict()

        state_dict = {
            "epoch": epoch, "args": vars(args), "stats": stats,
            "policy_state": policy_state, "decoder_state": decoder_state,
            "optimizer_state": optimizer.state_dict(),
            "vision_encoder_state": model.vision_encoder.state_dict(),
        }

        save_ckpt(state_dict, save_dir, "last.pth")
        if avg_val_a < best_val_loss:
            best_val_loss = avg_val_a; save_ckpt(state_dict, save_dir, "best.pth")
        if (epoch + 1) % args.save_interval == 0: save_ckpt(state_dict, save_dir, f"epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="checkpoints/policy")

    parser.add_argument("--base_policy", action="store_true")
    parser.add_argument("--decoder_ckpt", type=str, default=None)
    parser.add_argument("--train_decoder", action="store_true")

    parser.add_argument("--use_mip", action=argparse.BooleanOptionalAction, default=True, help="Enable MIP for the policy (default; disable with --no-use_mip)")

    parser.add_argument("--act_chunk", type=int, default=20, help="Predict future action chunk length K")

    parser.add_argument("--lambda_latent", type=float, default=1.0)
    parser.add_argument("--lambda_action", type=float, default=1.0)

    parser.add_argument("--frank_max_episodes", type=int, default=-1)
    parser.add_argument("--umi_max_episodes", type=int, default=0)
    parser.add_argument("--frank_real_max_episodes", type=int, default=0)
    parser.add_argument("--frank_sim_for_decoder", action="store_true")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--run_note", type=str, default="default")

    parser.add_argument("--action_type", type=str, default="joint")
    parser.add_argument("--robot_state_type", type=str, default="joint")

    parser.add_argument("--image_root", type=str, default="", help="Root directory containing image sequences")
    parser.add_argument("--camera_views", type=str, default="both", choices=["front", "overhead", "both"])
    parser.add_argument("--vision_feature_dim", type=int, default=64)
    parser.add_argument("--freeze_vision_encoder", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--vision_backbone", type=str, default="dinov2_vits14", choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for torch / numpy / random")

    args = parser.parse_args()
    train(args)


# MIP baseline (observation -> actions):
# python policy/train.py \
#     --data_root <policy_data_root> --image_root <data_root> \
#     --base_policy --frank_max_episodes 100 \
#     --save_dir checkpoints/policy/task1 --use_wandb

# GLAM-augmented policy (predict action latent, then decode to actions):
# python policy/train.py \
#     --data_root <policy_data_root> --image_root <data_root> \
#     --train_decoder --frank_max_episodes 100 --umi_max_episodes 400 \
#     --save_dir checkpoints/policy/task1 --use_wandb