import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler
import wandb
import os
import math
import random

from dataset import CombinedRoboticDataset
from model import GLAM


def vision_recon_loss(logits, target, image_mode):
    """Pick appropriate reconstruction loss based on image type.
    mask -> BCE with logits (binary target)
    masked_color / raw -> MSE after sigmoid (continuous [0,1] target)
    """
    if image_mode == 'mask':
        return F.binary_cross_entropy_with_logits(logits, target)
    else:
        return F.mse_loss(torch.sigmoid(logits), target)


class FixedRatioBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, frank_frac=0.25, drop_last=True, seed=0):
        if not (hasattr(dataset, "umi_len") and hasattr(dataset, "frank_len") and hasattr(dataset, "total_len")):
            raise AttributeError("FixedRatioBatchSampler requires CombinedRoboticDataset.")
        self.dataset = dataset; self.batch_size = int(batch_size)
        self.frank_frac = float(frank_frac); self.drop_last = bool(drop_last); self.seed = int(seed)
        self.umi_idx = list(range(0, dataset.umi_len))
        self.frank_idx = list(range(dataset.umi_len, dataset.total_len))
        if len(self.frank_idx) == 0: self.n_frank = 0; self.n_umi = self.batch_size
        elif len(self.umi_idx) == 0: self.n_umi = 0; self.n_frank = self.batch_size
        else:
            n_frank = int(round(self.batch_size * self.frank_frac))
            n_frank = max(1, min(self.batch_size - 1, n_frank))
            self.n_frank = n_frank; self.n_umi = self.batch_size - n_frank
        umi_batches = math.ceil(len(self.umi_idx) / max(1, self.n_umi)) if self.n_umi > 0 else 0
        frank_batches = math.ceil(len(self.frank_idx) / max(1, self.n_frank)) if self.n_frank > 0 else 0
        self.num_batches = max(1, umi_batches, frank_batches); self._epoch = 0

    def __len__(self): return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch); self._epoch += 1
        umi = self.umi_idx.copy(); frank = self.frank_idx.copy()
        rng.shuffle(umi); rng.shuffle(frank); up = 0; fp = 0
        for _ in range(self.num_batches):
            batch = []
            if self.n_umi > 0:
                if up + self.n_umi > len(umi): rng.shuffle(umi); up = 0
                batch.extend(umi[up:up + self.n_umi]); up += self.n_umi
            if self.n_frank > 0:
                if fp + self.n_frank > len(frank): rng.shuffle(frank); fp = 0
                batch.extend(frank[fp:fp + self.n_frank]); fp += self.n_frank
            rng.shuffle(batch); yield batch


def calc_masked_loss(pred, target, M, chunk_size):
    mse_elem = (pred - target) ** 2
    if chunk_size > 1:
        return (mse_elem.view(M.shape[0], -1).mean(dim=1, keepdim=True) * M).sum() / (M.sum() + 1e-6)
    else:
        return (mse_elem.mean(dim=1, keepdim=True) * M).sum() / (M.sum() + 1e-6)


def evaluate(model, val_loader, device, args, image_mode):
    model.eval()
    losses = {
        'total': 0, 'recon_obj': 0, 'recon_ee': 0, 'recon_rob': 0,
        'align': 0, 'recon_act': 0,
        'recon_act_rob': 0, 'recon_obj_rob': 0, 'recon_ee_rob': 0, 'recon_rob_rob': 0,
        'recon_rob_frank_sim': 0, 'recon_rob_frank_real': 0
    }
    count = 0
    count_sim = 0
    count_real = 0


    with torch.no_grad():
        for batch in val_loader:
            img_t = batch['img_mask_t'].to(device); img_tp1 = batch['img_mask_tp1'].to(device)
            ee_t = batch['ee_t'].to(device); ee_tp1 = batch['ee_tp1'].to(device)
            robot_s = batch['robot_state'].to(device); robot_s_tp1 = batch['robot_state_tp1'].to(device)
            robot_a = batch['robot_action'].to(device); M = batch['M'].to(device).unsqueeze(1)

            out = model(img_t, img_tp1, robot_s, robot_a, M, ee_t, ee_tp1)

            recon_obj = F.mse_loss(out['obj_tp1_pred'], out['obj_tp1_target'])
            recon_obj += vision_recon_loss(out['img_mask_tp1_recon_logits'], img_tp1, image_mode)
            if args.extra_auxilary_task:
                recon_obj += vision_recon_loss(out['img_mask_t_recon_logits'], img_t, image_mode)
                recon_obj += vision_recon_loss(out['img_mask_tp1_target_recon_logits'], img_tp1, image_mode)

            recon_ee = 0.0
            if args.use_idm_ee and out['ee_tp1_pred'] is not None:
                recon_ee = F.mse_loss(out['ee_tp1_pred'], out['ee_tp1_target'])
                
            recon_rob = 0.0
            if args.forward_robot and out['robot_s_tp1_pred'] is not None:
                recon_rob = calc_masked_loss(out['robot_s_tp1_pred'], robot_s_tp1, M, args.chunk_size)

                # Track sim and real robot data separately during validation
                source = batch.get('dataset_source')
                if source is not None:
                    M_sim = M * (source == 1).unsqueeze(1).float().to(device)
                    M_real = M * (source == 2).unsqueeze(1).float().to(device)
                    
                    if M_sim.sum() > 0:
                        losses['recon_rob_frank_sim'] += calc_masked_loss(out['robot_s_tp1_pred'], robot_s_tp1, M_sim, args.chunk_size).item()
                        count_sim += 1
                    if M_real.sum() > 0:
                        losses['recon_rob_frank_real'] += calc_masked_loss(out['robot_s_tp1_pred'], robot_s_tp1, M_real, args.chunk_size).item()
                        count_real += 1

            align_dist = F.mse_loss(out['mu_rob'], out['mu_idm'])
            recon_act = 0.0
            if args.recon_action and out['pred_action_recon'] is not None:
                recon_act = calc_masked_loss(out['pred_action_recon'], robot_a, M, args.chunk_size)
            recon_act_rob = 0.0
            if args.recon_action_rob and out.get('pred_action_rob') is not None:
                recon_act_rob = calc_masked_loss(out['pred_action_rob'], robot_a, M, args.chunk_size)

            recon_obj_rob = 0.0; recon_ee_rob = 0.0; recon_rob_rob = 0.0
            if args.rob_latent_forward and out.get('obj_tp1_pred_rob') is not None:
                recon_obj_rob = F.mse_loss(out['obj_tp1_pred_rob'], out['obj_tp1_target'])
                recon_obj_rob += vision_recon_loss(out['img_mask_tp1_rob_recon_logits'], img_tp1, image_mode)
                if args.use_idm_ee and out.get('ee_tp1_pred_rob') is not None:
                    recon_ee_rob = calc_masked_loss(out['ee_tp1_pred_rob'], out['ee_tp1_target'], M, args.chunk_size)
                if args.forward_robot and out.get('robot_s_tp1_pred_rob') is not None:
                    recon_rob_rob = calc_masked_loss(out['robot_s_tp1_pred_rob'], robot_s_tp1, M, args.chunk_size)

            total = recon_obj + float(recon_ee) + float(recon_rob) + align_dist \
                + args.act_recon_w * float(recon_act) \
                + args.act_recon_w_rob * float(recon_act_rob) \
                + args.rob_fwd_w * (float(recon_obj_rob) + float(recon_ee_rob) + 5.0 * float(recon_rob_rob))

            losses['total'] += total.item() if isinstance(total, torch.Tensor) else total
            losses['recon_obj'] += recon_obj.item() if isinstance(recon_obj, torch.Tensor) else recon_obj
            losses['recon_ee'] += float(recon_ee); losses['recon_rob'] += float(recon_rob)
            losses['align'] += align_dist.item(); losses['recon_act'] += float(recon_act)
            losses['recon_act_rob'] += float(recon_act_rob); losses['recon_obj_rob'] += float(recon_obj_rob)
            losses['recon_ee_rob'] += float(recon_ee_rob); losses['recon_rob_rob'] += float(recon_rob_rob)
            count += 1
            
    if count == 0: return losses

    for k in losses.keys():
        if k == 'recon_rob_frank_sim':
            losses[k] = losses[k] / count_sim if count_sim > 0 else 0.0
        elif k == 'recon_rob_frank_real':
            losses[k] = losses[k] / count_real if count_real > 0 else 0.0
        else:
            losses[k] = losses[k] / count
            
    return losses

def save_checkpoint(state, path): torch.save(state, path)

def save_dataset_indices(dataset, save_dir, split_name):
    lines = []
    if hasattr(dataset, 'umi_ds') and hasattr(dataset.umi_ds, 'files'):
        lines.extend([str(f) for f in dataset.umi_ds.files])
        
    if hasattr(dataset, 'frank_ds') and hasattr(dataset.frank_ds, 'files'):
        lines.extend([str(f) for f in dataset.frank_ds.files])

    if hasattr(dataset, 'frank_real_ds') and hasattr(dataset.frank_real_ds, 'files'):
        lines.extend([str(f) for f in dataset.frank_real_ds.files])
        
    if lines:
        save_path = os.path.join(save_dir, f"{split_name}_indices.txt")
        with open(save_path, 'w') as f:
            for item in lines: f.write(f"{item}\n")
        print(f"[INFO] Saved {len(lines)} {split_name} indices to {save_path}")

def parse_task_str(task_str):
    if task_str == 'all': return 'all'
    return [int(x) for x in task_str.split()]


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Determine image_mode
    image_mode = 'mask'
    if args.masked_color_image: image_mode = 'masked_color'
    elif args.raw_image: image_mode = 'raw'
    print(f"[CONFIG] image_mode: {image_mode}")

    # === Run Name Construction ===
    if args.use_idm_ee:
        if args.use_idm_ee_part: idm_mode = "_IDM-EEPart"
        elif args.idm_obj_only: idm_mode = "_IDM-ObjOnly"
        else: idm_mode = "_IDM-EE"
    else: idm_mode = ""
    rob_fwd = "_RobFwd" if args.forward_robot else ""
    act_recon_str = f"_ActRec{args.act_recon_w}" if args.recon_action else ""
    act_recon_str += f"_ActRecRob{args.act_recon_w_rob}" if args.recon_action_rob else ""
    act_recon_str += f"_RobFwdW{args.rob_fwd_w}" if args.rob_latent_forward else ""
    adapt_str = "_Adapt" if args.adaptive_subsample else "_Uniform"
    strict_rob = "_StrictRob" if args.strict_robot_encoder else ""
    view_str = "_" + "-".join(args.camera_views)

    vae_mode_str = ""
    if args.frozen_enc_dec: vae_mode_str += "_Frozen"
    elif args.finetune_enc_dec: vae_mode_str += "_Finetune"
    elif args.train_enc_dec: vae_mode_str += "_Scratch"
    if args.extra_auxilary_task: vae_mode_str += "_Aux"
    if image_mode != 'mask': vae_mode_str += f"_{image_mode}"

    limit_str = ""
    if args.frank_max_episodes > 0: limit_str += f"_F{args.frank_max_episodes}"
    if args.umi_max_episodes > 0: limit_str += f"_U{args.umi_max_episodes}"
    if args.frank_real_max_episodes > 0: limit_str += f"_FR{args.frank_real_max_episodes}"
    task_str = f"_T{args.frank_tasks}"
    if args.umi_tasks != args.frank_tasks: task_str += f"_UT{args.umi_tasks}"
    if args.frank_real_tasks != args.frank_tasks: task_str += f"_FRT{args.frank_real_tasks}"
    mix_str = f"_MixF{args.frank_batch_frac}" if args.frank_batch_frac >= 0 else ""
    subsample_str = f"_FS{args.frank_subsample}_US{args.umi_subsample}"
    chunk_str = f"_Chunk{args.chunk_size}" if args.chunk_size > 1 else ""

    run_name = (f"glam{view_str}{vae_mode_str}_bs{args.batch_size}_lat{args.latent_dim}_{args.action_type}_"
                f"{args.robot_state_type}{task_str}{limit_str}{adapt_str}{subsample_str}{chunk_str}{mix_str}"
                f"_align{args.align_w}{idm_mode}{rob_fwd}{strict_rob}{act_recon_str}")
    if args.robot_as_posterior: run_name += "_RobPost"

    ckpt_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    if args.use_wandb:
        wandb.init(project="glam_world_model", config=args, resume="allow", name=run_name)

    print("Initializing Datasets...")
    frank_tasks_parsed = parse_task_str(args.frank_tasks)
    umi_tasks_parsed = parse_task_str(args.umi_tasks)

    frank_real_tasks_parsed = parse_task_str(args.frank_real_tasks)

    train_dataset = CombinedRoboticDataset(
        umi_roots=args.umi_roots, frank_roots=args.frank_roots, frank_real_roots=args.frank_real_roots,
        umi_subsample=args.umi_subsample, frank_subsample=args.frank_subsample,
        split='train', val_ratio=args.val_ratio, action_type=args.action_type,
        robot_state_type=args.robot_state_type, camera_views=args.camera_views,
        adaptive_subsample=args.adaptive_subsample,
        frank_max_episodes=args.frank_max_episodes, umi_max_episodes=args.umi_max_episodes,
        frank_real_max_episodes=args.frank_real_max_episodes,
        frank_task_ids=frank_tasks_parsed, umi_task_ids=umi_tasks_parsed, frank_real_task_ids=frank_real_tasks_parsed,
        chunk_size=args.chunk_size, image_mode=image_mode)
    save_dataset_indices(train_dataset, ckpt_dir, "train")

    if args.frank_batch_frac < 0:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
    else:
        train_batch_sampler = FixedRatioBatchSampler(train_dataset, batch_size=args.batch_size, frank_frac=args.frank_batch_frac, drop_last=True, seed=args.seed)
        train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=4, pin_memory=True)

    val_dataset = CombinedRoboticDataset(
        umi_roots=args.umi_roots, frank_roots=args.frank_roots, frank_real_roots=args.frank_real_roots,
        umi_subsample=args.umi_subsample, frank_subsample=args.frank_subsample,
        split='val', val_ratio=args.val_ratio, action_type=args.action_type,
        robot_state_type=args.robot_state_type, camera_views=args.camera_views,
        normalizer=train_dataset.normalizer, adaptive_subsample=args.adaptive_subsample,
        frank_max_episodes=args.frank_max_episodes, umi_max_episodes=args.umi_max_episodes,
        frank_real_max_episodes=args.frank_real_max_episodes,
        frank_task_ids=frank_tasks_parsed, umi_task_ids=umi_tasks_parsed, frank_real_task_ids=frank_real_tasks_parsed,
        chunk_size=args.chunk_size, image_mode=image_mode)
    save_dataset_indices(val_dataset, ckpt_dir, "val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)

    model = GLAM(
        robot_state_dim=train_dataset.robot_state_dim, robot_action_dim=train_dataset.robot_action_dim,
        action_latent_dim=args.latent_dim, use_robot_forward=args.forward_robot,
        use_idm_ee=args.use_idm_ee, use_idm_ee_part=args.use_idm_ee_part,
        idm_obj_only=args.idm_obj_only, strict_robot_encoder=args.strict_robot_encoder,
        use_action_recon=args.recon_action, use_rob_action_recon=args.recon_action_rob,
        rob_latent_forward=args.rob_latent_forward, chunk_size=args.chunk_size,
        num_cameras=len(args.camera_views), image_mode=image_mode,
        ee_internal_dim=getattr(train_dataset, 'ee_internal_dim', 9)).to(device)
    
    print(f"[CONFIG] is_dual_arm={getattr(train_dataset, 'is_dual_arm', False)}, "
      f"ee_internal_dim={getattr(train_dataset, 'ee_internal_dim', 9)}, "
      f"robot_state_dim={train_dataset.robot_state_dim}, robot_action_dim={train_dataset.robot_action_dim}")

    if args.finetune_enc_dec or args.frozen_enc_dec:
        print(f"Loading VAE weights from {args.vae_ckpt}...")
        ckpt = torch.load(args.vae_ckpt, map_location=device)
        model.obj_encoder.load_state_dict(ckpt['encoder'])
        model.obj_decoder.load_state_dict(ckpt['decoder'])
    if args.frozen_enc_dec:
        print("Freezing Vision Encoder and Decoder...")
        for param in model.obj_encoder.parameters(): param.requires_grad = False
        for param in model.obj_decoder.parameters(): param.requires_grad = False
        model.obj_encoder.eval(); model.obj_decoder.eval()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)

    start_epoch, global_step, best_val_loss = 0, 0, float('inf')
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state']); optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1; global_step = ckpt.get('step', 0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))

    print("Start Training...")
    max_steps = args.max_steps; stop_training = False
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.frozen_enc_dec:
            model.obj_encoder.eval(); model.obj_decoder.eval()
        total_loss_sum = 0

        for batch in train_loader:
            img_t = batch['img_mask_t'].to(device); img_tp1 = batch['img_mask_tp1'].to(device)
            ee_t = batch['ee_t'].to(device); ee_tp1 = batch['ee_tp1'].to(device)
            robot_s = batch['robot_state'].to(device); robot_s_tp1 = batch['robot_state_tp1'].to(device)
            robot_a = batch['robot_action'].to(device); M = batch['M'].to(device).unsqueeze(1)

            out = model(img_t, img_tp1, robot_s, robot_a, M, ee_t, ee_tp1)

            # 1. Object Reconstruction
            recon_obj = F.mse_loss(out['obj_tp1_pred'], out['obj_tp1_target'])
            recon_obj += vision_recon_loss(out['img_mask_tp1_recon_logits'], img_tp1, image_mode)
            if args.extra_auxilary_task:
                recon_obj += vision_recon_loss(out['img_mask_t_recon_logits'], img_t, image_mode)
                recon_obj += vision_recon_loss(out['img_mask_tp1_target_recon_logits'], img_tp1, image_mode)

            # 2. EE Reconstruction
            recon_ee = torch.tensor(0.0, device=device)
            if args.use_idm_ee and out['ee_tp1_pred'] is not None:
                recon_ee = F.mse_loss(out['ee_tp1_pred'], out['ee_tp1_target'])

            # 3. Robot Forward
            recon_rob = torch.tensor(0.0, device=device)
            if args.forward_robot:
                recon_rob = calc_masked_loss(out['robot_s_tp1_pred'], robot_s_tp1, M, args.chunk_size)

            mu_idm, logvar_idm = out['mu_idm'], out['logvar_idm']
            mu_rob, logvar_rob = out['mu_rob'], out['logvar_rob']

            # 4 & 5. Latent Regulation
            if args.strict_robot_encoder:
                kl_idm_loss = torch.tensor(0.0, device=device)
                kl_align_elem = 0.5 * (logvar_rob - logvar_idm + (torch.exp(logvar_idm) + (mu_idm - mu_rob).pow(2)) / torch.exp(logvar_rob) - 1)
                if args.chunk_size > 1: align_loss = kl_align_elem.sum(-1).mean(-1).mean()
                else: align_loss = kl_align_elem.sum(1).mean()
            else:
                kl_idm_raw = -0.5 * (1 + logvar_idm - mu_idm.pow(2) - logvar_idm.exp())
                if args.chunk_size > 1: kl_idm_loss = kl_idm_raw.sum(-1).mean(-1).mean()
                else: kl_idm_loss = kl_idm_raw.sum(1).mean()

                kl_rob_loss = torch.tensor(0.0, device=device)
                kl_rob_raw = -0.5 * (1 + logvar_rob - mu_rob.pow(2) - logvar_rob.exp())
                if M.sum() > 0:
                    if args.chunk_size > 1: kl_rob_loss = (kl_rob_raw.sum(-1).mean(-1, keepdim=True) * M).sum() / (M.sum() + 1e-6)
                    else: kl_rob_loss = (kl_rob_raw.sum(1, keepdim=True) * M).sum() / (M.sum() + 1e-6)

                if args.robot_as_posterior:
                    target_mu = mu_rob.detach(); target_logvar = logvar_rob.detach()
                    kl_align_elem = 0.5 * (target_logvar - logvar_idm + (torch.exp(logvar_idm) + (mu_idm - target_mu).pow(2)) / torch.exp(target_logvar) - 1)
                else:
                    target_mu = mu_idm; target_logvar = logvar_idm
                    kl_align_elem = 0.5 * (target_logvar - logvar_rob + (torch.exp(logvar_rob) + (mu_rob - target_mu).pow(2)) / torch.exp(target_logvar) - 1)

                if args.chunk_size > 1: align_loss = (kl_align_elem.sum(-1).mean(-1, keepdim=True) * M).sum() / (M.sum() + 1e-6)
                else: align_loss = (kl_align_elem.sum(1, keepdim=True) * M).sum() / (M.sum() + 1e-6)

            # 6. IDM Action Recon
            recon_act = torch.tensor(0.0, device=device)
            if args.recon_action and out['pred_action_recon'] is not None:
                recon_act = calc_masked_loss(out['pred_action_recon'], robot_a, M, args.chunk_size)

            # 7. Robot Action Recon
            recon_act_rob = torch.tensor(0.0, device=device)
            if args.recon_action_rob and out.get('pred_action_rob') is not None:
                recon_act_rob = calc_masked_loss(out['pred_action_rob'], robot_a, M, args.chunk_size)

            # 8. Robot Latent Forward
            recon_obj_rob = torch.tensor(0.0, device=device)
            recon_ee_rob = torch.tensor(0.0, device=device)
            recon_rob_rob = torch.tensor(0.0, device=device)
            if args.rob_latent_forward and out.get('obj_tp1_pred_rob') is not None:
                recon_obj_rob = F.mse_loss(out['obj_tp1_pred_rob'], out['obj_tp1_target'])
                recon_obj_rob += vision_recon_loss(out['img_mask_tp1_rob_recon_logits'], img_tp1, image_mode)
                if args.use_idm_ee and out.get('ee_tp1_pred_rob') is not None:
                    recon_ee_rob = calc_masked_loss(out['ee_tp1_pred_rob'], out['ee_tp1_target'], M, args.chunk_size)
                if args.forward_robot and out.get('robot_s_tp1_pred_rob') is not None:
                    recon_rob_rob = calc_masked_loss(out['robot_s_tp1_pred_rob'], robot_s_tp1, M, args.chunk_size)

            loss = recon_obj + recon_ee + args.idm_kl * kl_idm_loss + args.idm_kl * kl_rob_loss \
                + args.align_w * align_loss + recon_rob \
                + args.act_recon_w * recon_act + args.act_recon_w_rob * recon_act_rob \
                + args.rob_fwd_w * (recon_obj_rob + recon_ee_rob + recon_rob_rob)

            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0); optimizer.step()

            total_loss_sum += loss.item(); global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                print(f"[INFO] Reached max_steps={max_steps}."); stop_training = True; break

            if global_step % 100 == 0:
                print(f"Step {global_step} | Obj: {recon_obj:.4f} | EE: {recon_ee:.4f} | Rob: {recon_rob:.4f} | Act: {recon_act:.4f} | Align: {align_loss:.4f} | KL: {kl_idm_loss:.4f}")
                if args.use_wandb:
                    log_dict = {
                        "train/loss": loss.item(),
                        "train/recon_obj": recon_obj.item() if isinstance(recon_obj, torch.Tensor) else recon_obj,
                        "train/recon_ee": recon_ee.item() if isinstance(recon_ee, torch.Tensor) else recon_ee,
                        "train/recon_rob": recon_rob.item() if isinstance(recon_rob, torch.Tensor) else recon_rob,
                        "train/recon_act": recon_act.item() if isinstance(recon_act, torch.Tensor) else recon_act,
                        "train/kl_idm": kl_idm_loss.item() if isinstance(kl_idm_loss, torch.Tensor) else kl_idm_loss,
                        "train/align": align_loss.item() if isinstance(align_loss, torch.Tensor) else align_loss,
                        "train/kl_rob": kl_rob_loss.item() if isinstance(kl_rob_loss, torch.Tensor) else kl_rob_loss,
                        "train/recon_act_rob": recon_act_rob.item() if isinstance(recon_act_rob, torch.Tensor) else recon_act_rob,
                        "train/recon_obj_rob": recon_obj_rob.item() if isinstance(recon_obj_rob, torch.Tensor) else recon_obj_rob,
                        "train/recon_ee_rob": recon_ee_rob.item() if isinstance(recon_ee_rob, torch.Tensor) else recon_ee_rob,
                        "train/recon_rob_rob": recon_rob_rob.item() if isinstance(recon_rob_rob, torch.Tensor) else recon_rob_rob,
                        "train/step": global_step
                    }
                    
                    # Track sim and real robot data separately
                    source = batch.get('dataset_source')
                    if source is not None and args.forward_robot:
                        M_sim = M * (source == 1).unsqueeze(1).float().to(device)
                        M_real = M * (source == 2).unsqueeze(1).float().to(device)
                        
                        if M_sim.sum() > 0:
                            log_dict["train/recon_rob_frank_sim"] = calc_masked_loss(out['robot_s_tp1_pred'], robot_s_tp1, M_sim, args.chunk_size).item()
                        if M_real.sum() > 0:
                            log_dict["train/recon_rob_frank_real"] = calc_masked_loss(out['robot_s_tp1_pred'], robot_s_tp1, M_real, args.chunk_size).item()
                            
                    wandb.log(log_dict)

        if stop_training: break

        if (epoch + 1) % args.eval_freq == 0:
            val_metrics = evaluate(model, val_loader, device, args, image_mode)
            val_loss = val_metrics['total']
            state_dict = {
                'epoch': epoch, 'step': global_step,
                'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
                'stats': train_dataset.normalizer.stats, 'best_val_loss': best_val_loss, 'args': vars(args)}
            if val_loss < best_val_loss:
                best_val_loss = val_loss; state_dict['best_val_loss'] = best_val_loss
                save_checkpoint(state_dict, os.path.join(ckpt_dir, "best_val.pth"))
            save_checkpoint(state_dict, os.path.join(ckpt_dir, "latest.pth"))
            if (epoch + 1) % 10 == 0:
                save_checkpoint(state_dict, os.path.join(ckpt_dir, f"glam_epoch_{epoch+1}.pth"))
            print(f">>> Val Epoch {epoch}: Loss {val_loss:.4f}")
            if args.use_wandb:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
                wandb.log({"val/epoch": epoch})

    print("[INFO] Running final validation before exit...")
    val_metrics = evaluate(model, val_loader, device, args, image_mode)
    val_loss = val_metrics['total']
    state_dict = {
        'epoch': epoch, 'step': global_step,
        'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(),
        'stats': train_dataset.normalizer.stats, 'best_val_loss': best_val_loss, 'args': vars(args)}
    save_checkpoint(state_dict, os.path.join(ckpt_dir, "latest.pth"))
    if val_loss < best_val_loss:
        print("[INFO] Final validation is the best so far. Updating best_val.pth.")
        best_val_loss = val_loss; state_dict['best_val_loss'] = best_val_loss
        save_checkpoint(state_dict, os.path.join(ckpt_dir, "best_val.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Defaults below are the configuration used for the paper experiments;
    # boolean components can be disabled with --no-<flag> for ablations.
    parser.add_argument("--umi_roots", nargs='+', required=True)
    parser.add_argument("--frank_roots", nargs='+', required=True)
    parser.add_argument("--epochs", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--align_w", type=float, default=1.0)
    parser.add_argument("--idm_kl", type=float, default=1e-3)
    parser.add_argument("--chunk_size", type=int, default=1)
    parser.add_argument("--frank_batch_frac", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--umi_subsample", type=int, default=1)
    parser.add_argument("--frank_subsample", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="joint", choices=["joint", "ee", "ee_delta"])
    parser.add_argument("--robot_state_type", type=str, default="ee", choices=["joint", "ee"])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--camera_views", nargs='+', default=["overhead", "front"])
    parser.add_argument("--frank_tasks", type=str, default="all")
    parser.add_argument("--umi_tasks", type=str, default="all")
    parser.add_argument("--forward_robot", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_idm_ee", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_idm_ee_part", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--idm_obj_only", action="store_true")
    parser.add_argument("--adaptive_subsample", action="store_true")
    parser.add_argument("--frank_max_episodes", type=int, default=-1)
    parser.add_argument("--umi_max_episodes", type=int, default=-1)
    parser.add_argument("--strict_robot_encoder", action="store_true")
    parser.add_argument("--recon_action", action="store_true")
    parser.add_argument("--act_recon_w", type=float, default=1)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--save_dir", type=str, default="checkpoints/glam")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--recon_action_rob", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--act_recon_w_rob", type=float, default=1.0)
    parser.add_argument("--rob_latent_forward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rob_fwd_w", type=float, default=1.0)
    parser.add_argument("--robot_as_posterior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_steps", type=int, default=-1)
    # Vision encoder/decoder
    parser.add_argument("--train_enc_dec", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune_enc_dec", action="store_true")
    parser.add_argument("--frozen_enc_dec", action="store_true")
    parser.add_argument("--extra_auxilary_task", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vae_ckpt", type=str, default="")
    # Image mode
    parser.add_argument("--masked_color_image", action="store_true", help="Use mask * raw RGB as vision input")
    parser.add_argument("--raw_image", action="store_true", help="Use raw RGB image as vision input")

    parser.add_argument("--frank_real_roots", nargs='+', default=None)
    parser.add_argument("--frank_real_max_episodes", type=int, default=-1)
    parser.add_argument("--frank_real_tasks", type=str, default="all")

    args = parser.parse_args()
    train(args)


# Example (model/loss configuration is on by default; add --frank_real_roots to
# co-train with real-robot demonstrations):
# python glam/train.py \
#     --umi_roots <data_root>/umi/ --frank_roots <data_root>/frank/ \
#     --umi_tasks 1 --frank_tasks 1 \
#     --umi_max_episodes 400 --frank_max_episodes 100 \
#     --latent_dim 8 --max_steps 15000 --use_wandb