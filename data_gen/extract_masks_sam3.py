"""
Local SAM 3 Mask Extraction Script (Dual Mode & Range Index)
=======================================================
Includes:
1. Video Tracking / Per-Image Mode.
2. Inference state reset for robust batch processing.
3. Connected Component Analysis.
4. Sequence range filtering (--index).
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import numpy as np
from pathlib import Path
import cv2
import sys
import os
from tqdm import tqdm

try:
    from ultralytics.models.sam import SAM3SemanticPredictor, SAM3VideoSemanticPredictor
except ImportError:
    print("[ERROR] Please upgrade ultralytics: pip install -U ultralytics")
    sys.exit(1)

# ============================================
# Utils
# ============================================
def parse_int_safe(x):
    try: return int(x)
    except Exception: return None

def frame_sort_key(img_path: Path):
    stem = img_path.stem
    stem_int = parse_int_safe(stem)
    if stem_int is not None: return (0, stem_int, stem)
    return (1, stem)

def sequence_sort_key(item):
    episode_dir, emb = item
    parts = episode_dir.parts
    task_name = episode_dir.parent.parent.name if len(parts) >= 3 else ""
    status_name = episode_dir.parent.name if len(parts) >= 2 else ""
    episode_name = episode_dir.name
    episode_int = parse_int_safe(episode_name)
    return (emb, task_name, status_name, (0, episode_int, episode_name) if episode_int is not None else (1, episode_name))

def find_sequence_dirs(data_root, embodiments=None):
    data_root = Path(data_root)
    sequence_dirs = []
    embodiments = embodiments or ["umi", "frank"]

    for emb in embodiments:
        emb_root = data_root / emb
        if not emb_root.exists(): continue
        for task_dir in sorted(emb_root.glob("task_*"), key=lambda p: p.name):
            for status_dir in sorted(task_dir.glob("*"), key=lambda p: p.name):
                for episode_dir in sorted(status_dir.glob("*"), key=lambda p: p.name):
                    if (episode_dir / "front_frames").exists() or (episode_dir / "overhead_frames").exists():
                        sequence_dirs.append((episode_dir, emb))
    return sorted(sequence_dirs, key=sequence_sort_key)

# ============================================
# Post-Processing: Keep Largest Component
# ============================================
def keep_largest_component(mask):
    """
    Removes small artifacts/noise. Keeps only the largest connected chunk.
    """
    if np.max(mask) == 0:
        return mask
        
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return mask
        
    areas = stats[1:, cv2.CC_STAT_AREA] 
    largest_label = np.argmax(areas) + 1 
    
    cleaned_mask = np.zeros_like(mask)
    cleaned_mask[labels == largest_label] = 255
    
    return cleaned_mask

# ============================================
# Post-Processing: Filter Small Artifacts Only
# ============================================
def filter_small_components(mask, min_area_ratio=0.15):

    if np.max(mask) == 0:
        return mask
        
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1:
        return mask
        
    areas = stats[1:, cv2.CC_STAT_AREA] 
    max_area = np.max(areas)
    
    cleaned_mask = np.zeros_like(mask)
    
    for i, area in enumerate(areas):
        if area >= max_area * min_area_ratio:
            cleaned_mask[labels == (i + 1)] = 255
            
    return cleaned_mask

# ============================================
# Core Pipeline
# ============================================
def process_sequence(
    sequence_dir, camera_views, stats,
    predictor, text_prompt="blue cube",
    save_overlay=False, use_video_tracking=False,
    multi_object=False
):
    sequence_dir = Path(sequence_dir)
    # Single-object prompts keep only the largest component (removes stray
    # detections); multi-object prompts must keep every sizable component.
    postprocess = filter_small_components if multi_object else keep_largest_component

    for view in camera_views:
        frames_dir = sequence_dir / f"{view}_frames"
        mask_dir = sequence_dir / f"{view}_frames_mask"

        if not frames_dir.exists(): continue
        mask_dir.mkdir(parents=True, exist_ok=True)

        overlay_dir = sequence_dir / f"{view}_frames_mask_overlay" if save_overlay else None
        if overlay_dir: overlay_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg")), key=frame_sort_key)
        if not image_files: continue

        # ---------------------------------------------------------
        # MODE 1: Video Tracking (Temporal Memory)
        # ---------------------------------------------------------
        if use_video_tracking:
            temp_video_path = str(sequence_dir / f"temp_track_{view}.mp4")
            first_frame = cv2.imread(str(image_files[0]))
            if first_frame is None: continue
            
            h, w = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (w, h))
            
            for img_path in image_files:
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is not None: video_writer.write(img_bgr)
            video_writer.release()

            try:
                if hasattr(predictor, 'inference_state'):
                    predictor.inference_state = {}
                
                results = predictor(source=temp_video_path, text=[text_prompt], stream=True)
                
                for img_path, res in zip(image_files, results):
                    stats["total_frames"] += 1
                    frame_name = img_path.stem
                    
                    image_bgr = cv2.imread(str(img_path))
                    final_mask = np.zeros((h, w), dtype=np.uint8)
                    
                    if res.masks is not None:
                        masks_tensor = res.masks.data.cpu().numpy()
                        final_mask = np.any(masks_tensor > 0.5, axis=0).astype(np.uint8) * 255
                        final_mask = postprocess(final_mask)

                        if np.max(final_mask) > 0:
                            stats["success"] += 1
                        else:
                            stats["miss"] += 1
                    else:
                        stats["miss"] += 1

                    cv2.imwrite(str(mask_dir / f"{frame_name}.png"), final_mask)
                    stats["saved_masks"] += 1

                    if overlay_dir is not None:
                        overlay = image_bgr.copy()
                        overlay[final_mask > 0] = (overlay[final_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5)
                        cv2.imwrite(str(overlay_dir / f"{frame_name}.png"), overlay.astype(np.uint8))
            
            finally:
                if os.path.exists(temp_video_path):
                    os.remove(temp_video_path)

        # ---------------------------------------------------------
        # MODE 2: Per-Image Prediction
        # ---------------------------------------------------------
        else:
            frame_iter = tqdm(image_files, desc=f"{view}", leave=False)
            
            for img_path in frame_iter:
                stats["total_frames"] += 1
                frame_name = img_path.stem
                
                predictor.set_image(str(img_path))
                results = predictor(text=[text_prompt])
                
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    stats["read_fail"] += 1
                    continue
                    
                final_mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
                
                if results and len(results) > 0 and results[0].masks is not None:
                    masks_tensor = results[0].masks.data.cpu().numpy()
                    final_mask = np.any(masks_tensor > 0.5, axis=0).astype(np.uint8) * 255
                    final_mask = postprocess(final_mask)
                    if np.max(final_mask) > 0:
                        stats["success"] += 1
                    else:
                        stats["miss"] += 1
                else:
                    stats["miss"] += 1

                cv2.imwrite(str(mask_dir / f"{frame_name}.png"), final_mask)
                stats["saved_masks"] += 1

                if overlay_dir is not None:
                    overlay = image_bgr.copy()
                    overlay[final_mask > 0] = (overlay[final_mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5)
                    cv2.imwrite(str(overlay_dir / f"{frame_name}.png"), overlay.astype(np.uint8))

# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--embodiments", nargs="+", default=["umi", "frank"])
    parser.add_argument("--camera_views", nargs="+", default=["front", "overhead"])
    parser.add_argument("--text_prompt", type=str, default="blue cube")
    parser.add_argument("--model_path", type=str, default="sam3.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--save_overlay", action="store_true")
    parser.add_argument("--max_sequences", type=int, default=-1)
    parser.add_argument("--use_video_tracking", action="store_true")
    parser.add_argument("--multi_object", action="store_true",
                        help="Keep every mask component >= 15%% of the largest one. Required when the "
                             "text prompt names multiple objects (e.g. stack three: 'blue cube and "
                             "purple cube'); by default only the largest component is kept.")
    
    parser.add_argument("--index", nargs="+", type=int, help="Sequence index range (e.g., --index 5 400 or --index 10)")

    args = parser.parse_args()

    sequence_dirs = find_sequence_dirs(args.data_root, args.embodiments)
    
    if args.index is not None:
        if len(args.index) == 1:
            start_idx = args.index[0]
            end_idx = args.index[0]
        else:
            start_idx = args.index[0]
            end_idx = args.index[1]
            
        filtered_dirs = []
        for seq_dir, emb in sequence_dirs:
            seq_int = parse_int_safe(seq_dir.name)
            if seq_int is not None and start_idx <= seq_int <= end_idx:
                filtered_dirs.append((seq_dir, emb))
        sequence_dirs = filtered_dirs

    if args.max_sequences > 0: sequence_dirs = sequence_dirs[:args.max_sequences]

    if not sequence_dirs:
        print("[WARNING] No sequences found for the given criteria (embodiments, index range).")
        return

    overrides = dict(
        conf=args.confidence, 
        task="segment", 
        mode="predict", 
        model=args.model_path, 
        half=True,       
        imgsz=644,       
        save=False,      
        verbose=False    
    )

    if args.use_video_tracking:
        print(f"Loading Local SAM 3 [VIDEO TRACKER] from {args.model_path}...")
        predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    else:
        print(f"Loading Local SAM 3 [IMAGE PREDICTOR] from {args.model_path}...")
        predictor = SAM3SemanticPredictor(overrides=overrides)

    print("Model loaded successfully!")

    stats = {"total_frames": 0, "saved_masks": 0, "read_fail": 0, "success": 0, "miss": 0}

    outer_bar = tqdm(sequence_dirs, desc="Processing sequences", leave=True)
    for seq_dir, emb_name in outer_bar:
        outer_bar.set_postfix_str(f"{emb_name} | {seq_dir.name}")
        process_sequence(
            sequence_dir=seq_dir, camera_views=args.camera_views, stats=stats,
            predictor=predictor, text_prompt=args.text_prompt,
            save_overlay=args.save_overlay, use_video_tracking=args.use_video_tracking,
            multi_object=args.multi_object,
        )

    print("\nExtraction Complete.")
    print(f"Total frames   : {stats['total_frames']}")
    print(f"SAM 3 Success  : {stats['success']}")
    print(f"SAM 3 Missed   : {stats['miss']}")

if __name__ == "__main__":
    main()


