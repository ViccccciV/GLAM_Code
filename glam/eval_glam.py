"""Run the full GLAM world-model evaluation pipeline in one command.

For every GLAM checkpoint it:
  1) trains an action decoder on the frozen latents (train_action_decoder.py)
  2) generates action predictions on held-out episodes
     (generate_val_predictions.py) -> val_predictions_{frank,umi,frank_real}.pkl

The pkl files can then be replayed in the MuJoCo simulation with
sim_eval/eval_predictions_in_sim.py and sim_eval/eval_predictions_in_sim_umi.py.

Model configuration (latent dim, camera views, chunk size, image mode, ...) is
read from each checkpoint automatically.
"""
import argparse
import os
import sys
import subprocess
import torch
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any


def _resolve_dir(base: Path, maybe_relative: str) -> Path:
    p = Path(maybe_relative)
    return p if p.is_absolute() else (base / p)


def get_ckpt_args(ckpt_path: Path) -> dict:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        if 'args' in ckpt: return ckpt['args']
    except Exception as e: print(f"[WARN] Could not load args from {ckpt_path}: {e}")
    return {}


def run_cmd(name: str, cmd: List[str], cwd: Optional[Path] = None, dry_run: bool = False, stop_on_error: bool = False, env: Optional[dict] = None) -> Dict[str, Any]:
    cmd_str = " ".join([str(x) for x in cmd])
    print("\n" + "=" * 120); print(f"[STEP] {name}"); print("[RUN ]", cmd_str); print("=" * 120)

    if dry_run: return {"ok": True, "returncode": 0, "name": name, "cmd": cmd}
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, check=False, text=True)

    ok = (p.returncode == 0)
    if ok: print(f"[OK  ] {name} finished with returncode=0")
    else:
        print(f"[FAIL] {name} failed with returncode={p.returncode}")
        if stop_on_error: raise RuntimeError(f"Step '{name}' failed. Stop on error is enabled.")
    return {"ok": ok, "returncode": p.returncode, "name": name, "cmd": cmd}


def main():
    parser = argparse.ArgumentParser("eval_glam.py")
    parser.add_argument("--glam_ckpts", nargs="+", default=None, help="Explicit list of GLAM checkpoint paths")
    parser.add_argument("--glam_ckpts_base_dir", type=str, default=None, help="Evaluate every best_val.pth found under this directory")
    parser.add_argument("--umi_roots", nargs="+", required=True)
    parser.add_argument("--frank_roots", nargs="+", required=True)
    parser.add_argument("--frank_real_roots", nargs="+", default=None)
    parser.add_argument("--encoder_source", type=str, default="idm", choices=["robot", "idm"])
    parser.add_argument("--target_action_type", type=str, default="joint", choices=["joint", "ee", "ee_delta"])

    parser.add_argument("--umi_subsample", type=int, default=1)
    parser.add_argument("--frank_subsample", type=int, default=1)
    parser.add_argument("--frank_real_subsample", type=int, default=1)
    parser.add_argument("--frank_max_episodes", type=int, default=-1)
    parser.add_argument("--frank_real_max_episodes", type=int, default=-1)
    parser.add_argument("--umi_max_episodes", type=int, default=-1)
    parser.add_argument("--frank_tasks", type=str, default="all")
    parser.add_argument("--frank_real_tasks", type=str, default="all")
    parser.add_argument("--umi_tasks", type=str, default="all")
    parser.add_argument("--umi_task_ids", type=str, default="all")

    parser.add_argument("--project_root", type=str, default=None)
    parser.add_argument("--decoder_save_dir", type=str, default="wm_eval/action_decoder_ckpts")
    parser.add_argument("--inference_out_dir", type=str, default="wm_eval/inference_results")
    parser.add_argument("--decoder_epochs", type=int, default=10)

    parser.add_argument("--frank_num_episodes", type=int, default=10, help="Held-out Frank episodes to run inference on")
    parser.add_argument("--umi_base_dir", type=str, default=None, help="UMI success dir (e.g. <data_root>/umi/task_01/success/)")
    parser.add_argument("--episode_ids", nargs="+", type=int, default=None, help="UMI episode ids to run inference on")

    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    parser.add_argument("--skip_train_decoder", action="store_true")
    parser.add_argument("--skip_frank_infer", action="store_true")
    parser.add_argument("--skip_frank_real_infer", action="store_true")
    parser.add_argument("--skip_umi_infer", action="store_true")
    parser.add_argument("--decoder_ckpt", type=str, default=None, help="Reuse an existing decoder instead of training one")
    parser.add_argument("--mplconfigdir", type=str, default="/tmp/matplotlib")

    args = parser.parse_args()

    glam_ckpt_list: List[Path] = []
    if args.glam_ckpts: glam_ckpt_list = [Path(p).resolve() for p in args.glam_ckpts]
    elif args.glam_ckpts_base_dir:
        base_dir = Path(args.glam_ckpts_base_dir).resolve()
        glam_ckpt_list = sorted(base_dir.glob("**/best_val.pth"))
        if not glam_ckpt_list: raise RuntimeError(f"No best_val.pth found under {base_dir}")
    else: raise ValueError("You must provide either --glam_ckpts or --glam_ckpts_base_dir.")

    py = Path(args.python)
    if not py.exists(): py = Path(sys.executable)
    this_dir = Path(__file__).resolve().parent
    project_root = Path(args.project_root).resolve() if args.project_root else this_dir.parent

    decoder_root = _resolve_dir(project_root, args.decoder_save_dir)
    infer_root = _resolve_dir(project_root, args.inference_out_dir)

    train_dec_py = this_dir / "train_action_decoder.py"
    infer_py = this_dir / "generate_val_predictions.py"

    env = os.environ.copy()
    if args.mplconfigdir:
        Path(args.mplconfigdir).mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = args.mplconfigdir

    total = len(glam_ckpt_list)
    for i, glam_ckpt in enumerate(glam_ckpt_list):
        if not glam_ckpt.exists(): continue
        print(f"\n{'#'*80}\n[{i+1}/{total}] Processing: {glam_ckpt.name}\n{'#'*80}")

        ckpt_args = get_ckpt_args(glam_ckpt)
        us = ckpt_args.get("umi_subsample", args.umi_subsample)
        fs = ckpt_args.get("frank_subsample", args.frank_subsample)
        frs = ckpt_args.get("frank_real_subsample", args.frank_real_subsample)
        f_max = ckpt_args.get("frank_max_episodes", args.frank_max_episodes)
        fr_max = ckpt_args.get("frank_real_max_episodes", args.frank_real_max_episodes)
        u_max = ckpt_args.get("umi_max_episodes", args.umi_max_episodes)
        chunk_size = ckpt_args.get("chunk_size", 1)

        model_name = glam_ckpt.parent.name
        suffix = f"_{glam_ckpt.stem}"
        temp_dir_name = f"temp{suffix}"

        dec_run_name_base = f"ActDec_{args.target_action_type}_from_{args.encoder_source}__for__{model_name}"
        final_dec_dir = decoder_root / f"{dec_run_name_base}{suffix}"
        current_decoder_ckpt = final_dec_dir / "best_decoder.pth"

        if args.decoder_ckpt: current_decoder_ckpt = Path(args.decoder_ckpt).resolve()
        else:
            # ------------------ 1. TRAIN DECODER ------------------
            if not args.skip_train_decoder:
                temp_dec_out = decoder_root / temp_dir_name
                cmd = [
                    str(py), str(train_dec_py),
                    "--umi_roots", *args.umi_roots, "--frank_roots", *args.frank_roots,
                    "--encoder_source", str(args.encoder_source), "--target_action_type", str(args.target_action_type),
                    "--glam_ckpt", str(glam_ckpt), "--save_dir", str(temp_dec_out),
                    "--umi_subsample", str(us), "--frank_subsample", str(fs),
                    "--frank_tasks", str(args.frank_tasks), "--umi_task_ids", str(args.umi_task_ids),
                    "--frank_max_episodes", str(f_max), "--umi_max_episodes", str(u_max),
                    "--epochs", str(args.decoder_epochs),
                    "--chunk_size", str(chunk_size),
                ]
                if args.frank_real_roots:
                    cmd.extend(["--frank_real_roots", *args.frank_real_roots,
                                "--frank_real_max_episodes", str(fr_max),
                                "--frank_real_tasks", str(args.frank_real_tasks),
                                "--frank_real_subsample", str(frs)])
                res = run_cmd("train_decoder", cmd, cwd=None, dry_run=args.dry_run, stop_on_error=args.stop_on_error, env=env)
                if res['ok'] and not args.dry_run:
                    src = temp_dec_out / dec_run_name_base
                    if src.exists():
                        if final_dec_dir.exists(): shutil.rmtree(final_dec_dir)
                        shutil.move(str(src), str(final_dec_dir)); shutil.rmtree(temp_dec_out)

        if not current_decoder_ckpt.exists() and not args.dry_run: continue

        final_infer_dir = infer_root / f"{model_name}{suffix}"; temp_infer_out = infer_root / temp_dir_name

        # ------------------ 2A. INFER FRANK ------------------
        if not args.skip_frank_infer:
            cmd = [
                str(py), str(infer_py), "--glam_ckpt", str(glam_ckpt), "--decoder_ckpt", str(current_decoder_ckpt),
                "--out_dir", str(temp_infer_out), "--encoder_source", str(args.encoder_source), "--inference_source", "frank",
                "--frank_subsample", str(fs), "--umi_subsample", str(us), "--num_episodes", str(args.frank_num_episodes),
                "--chunk_size", str(chunk_size),
            ]
            run_cmd("infer_frank", cmd, cwd=None, dry_run=args.dry_run, stop_on_error=args.stop_on_error, env=env)

        # ------------------ 2B. INFER FRANK REAL ------------------
        if not args.skip_frank_real_infer and args.frank_real_roots:
            cmd = [
                str(py), str(infer_py), "--glam_ckpt", str(glam_ckpt), "--decoder_ckpt", str(current_decoder_ckpt),
                "--out_dir", str(temp_infer_out), "--encoder_source", str(args.encoder_source), "--inference_source", "frank_real",
                "--frank_subsample", str(frs), "--umi_subsample", str(us), "--num_episodes", str(args.frank_num_episodes),
                "--chunk_size", str(chunk_size),
            ]
            run_cmd("infer_frank_real", cmd, cwd=None, dry_run=args.dry_run, stop_on_error=args.stop_on_error, env=env)

        # ------------------ 2C. INFER UMI ------------------
        if not args.skip_umi_infer:
            if args.umi_base_dir and args.episode_ids:
                cmd = [
                    str(py), str(infer_py), "--glam_ckpt", str(glam_ckpt), "--decoder_ckpt", str(current_decoder_ckpt),
                    "--out_dir", str(temp_infer_out), "--encoder_source", str(args.encoder_source), "--inference_source", "umi",
                    "--frank_subsample", str(fs), "--umi_subsample", str(us), "--umi_base_dir", str(args.umi_base_dir),
                    "--episode_ids", *[str(x) for x in args.episode_ids], "--chunk_size", str(chunk_size),
                ]
                run_cmd("infer_umi", cmd, cwd=None, dry_run=args.dry_run, stop_on_error=args.stop_on_error, env=env)

        if (not args.dry_run) and temp_infer_out.exists():
            src = temp_infer_out / model_name
            if src.exists():
                if final_infer_dir.exists(): shutil.rmtree(final_infer_dir)
                shutil.move(str(src), str(final_infer_dir))
            shutil.rmtree(temp_infer_out)

    print("\n" + "=" * 120 + "\nALL DONE\n" + "=" * 120)


if __name__ == "__main__":
    main()


# Example:
# python glam/eval_glam.py \
#     --glam_ckpts checkpoints/glam/<run_name>/best_val.pth \
#     --umi_roots data/stack_two/umi/ --frank_roots data/stack_two/frank/ \
#     --umi_tasks 1 --frank_tasks 1 --umi_task_ids 1 \
#     --umi_base_dir data/stack_two/umi/task_01/success/ --episode_ids 0 1 2 3 4
