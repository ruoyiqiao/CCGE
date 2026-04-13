#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task
# This script runs PPO training for the singulation and grasp task
export CUDA_VISIBLE_DEVICES=0

python src/train.py \
    headless=True \
    env_info=True \
    num_envs=2048 \
    num_objects=5 \
    num_objects_per_env=5 \
    graphics_device_id=0 \
    split='train' \
    cluster=0 \
    task=XArmLeapHandFunctionalManipulationUnderarm \
    train=XArmLeapHandFunctionalManipulationUnderarmPPO \
    task.env.enableContactSensors=False \
    reward_type=target+bonus+success+reach+contact_coverage+energy_reach \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    physics_engine=physx \
    +task.env.use_center_collision=False \
    +task.env.maskBackfacePoints=True \
    +task.env.maskPalmInwardPoints=True \
    +task.env.useForceDirectionBins=False \
    +task.env.useNormalInClustering=False \
    +task.env.randomObjectPosition=True \
    +task.env.randomObjectPositionOnReset=True \
    +task.env.stateFeatureDim=192 \
    +task.env.stateType=hash \
    +task.env.numKeyStates=32 \
    +task.env.stateIncludeGoal=True \
    +task.env.stateRunningMaxMode=global \
    --cfg_train=XArmLeapHandFunctionalManipulationUnderarmPPO \
    --seed=1 \
    --exp_name='PPO' \
    --logdir='leap_singulation' \
    --run_device_id=0 \
    --web_visualizer_port=-1 \
    # --con \
