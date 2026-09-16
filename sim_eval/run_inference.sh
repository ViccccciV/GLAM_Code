#!/bin/bash
# Evaluate trained BC policies in the MuJoCo Frank simulation.
#
# The best BC checkpoint is not known a priori, so this sweeps several epochs
# for each run. Every (run, checkpoint) pair is evaluated with the
# randomized-initial-state protocol: num_groups x trials_per_group episodes,
# reported as overall success rate with a Wilson 95% confidence interval.
#
# Run from the sim_eval/ directory with the `glam_eval` conda environment.

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

TASK=1                                # 1 = stack two, 2 = stack three
CKPT_ROOT="../checkpoints/policy/task${TASK}"
OUT_ROOT="inference_results/task0${TASK}"

# Checkpoints to sweep (without .pth suffix)
TARGET_CKPTS=("epoch_60" "epoch_80" "epoch_100" "epoch_120" "epoch_140")

# Checkpoint run folders to evaluate (as produced by policy/train.py), e.g.
#   glam_CustomData_BasePolicy_MLP_PolMIP_VisBoth_vits14_FT_ActC20_..._seed42_default
#   glam_CustomData_LatentPolicy_TrainDec_PolMIP_VisBoth_vits14_FT_ActC20_..._seed42_default
RUNS=(
    "glam_CustomData_BasePolicy_MLP_PolMIP_VisBoth_vits14_FT_ActC20_bs64_Lz1.0_La1.0_seed42_default"
    "glam_CustomData_LatentPolicy_TrainDec_PolMIP_VisBoth_vits14_FT_ActC20_bs64_Lz1.0_La1.0_seed42_default"
)

for RUN in "${RUNS[@]}"; do
    for CKPT in "${TARGET_CKPTS[@]}"; do
        echo "================================================================"
        echo ">> Evaluating: ${RUN} / ${CKPT}.pth"
        echo "================================================================"
        python inference_mujoco.py \
            --task ${TASK} \
            --use_random_init --num_groups 5 --trials_per_group 10 \
            --save_video --early_stop --replan_every 20 --seed 0 \
            --output_dir "${OUT_ROOT}/${RUN}/${CKPT}" \
            --ckpt_path "${CKPT_ROOT}/${RUN}/${CKPT}.pth"
    done
done

echo "All evaluations completed."
