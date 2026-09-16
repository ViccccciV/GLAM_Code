#!/bin/bash
# Replay GLAM world-model predictions (from glam/eval_glam.py) in the MuJoCo
# Frank simulation: GT-vs-prediction videos for Frank episodes, prediction-only
# videos for UMI episodes, plus success logs.
#
# Run from the sim_eval/ directory with the `glam_eval` conda environment.

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
set -e

TASK=1                                 # 1 = stack two, 2 = stack three
DATA_ROOT="../data/stack_two"
PRED_DIR="../wm_eval/inference_results/<run_name>_best_val"   # output folder of glam/eval_glam.py

echo "Replaying UMI predictions..."
python eval_predictions_in_sim_umi.py \
    --pred_file "${PRED_DIR}/val_predictions_umi.pkl" \
    --umi_data_root "${DATA_ROOT}/umi/task_0${TASK}/success/" \
    --task ${TASK} \
    --repeat_list 1 \
    --render_each_action \
    --check_success \
    --replan_every 1

echo "Replaying Frank predictions..."
python eval_predictions_in_sim.py \
    --pred_file "${PRED_DIR}/val_predictions_frank.pkl" \
    --data_root "${DATA_ROOT}/frank/task_0${TASK}/" \
    --task ${TASK} \
    --repeat_list 1 \
    --render_each_action \
    --check_success \
    --replan_every 1

echo "All evaluations completed."
