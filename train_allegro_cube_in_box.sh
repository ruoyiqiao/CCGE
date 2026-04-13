#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task
# This script runs PPO training for the singulation and grasp task

export CUDA_VISIBLE_DEVICES=0

python src/train.py \
    headless=True \
    env_info=True \
    num_envs=2048 \
    num_objects=1 \
    num_objects_per_env=1 \
    graphics_device_id=0 \
    split='train' \
    cluster=0 \
    task=XArmAllegroHandCubeInBox \
    train=XArmAllegroHandCubeInBoxPPO \
    task.env.enableContactSensors=False \
    reward_type=target+bonus+success+energy_reach+contact_coverage \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    physics_engine=physx \
    task.env.enableDebugVis=False \
    task.env.enableRenderedPointCloud=False \
    task.env.enableRenderedPointCloudTargetMask=False \
    +task.env.use_center_collision=False \
    +task.env.maskBackfacePoints=True \
    +task.env.maskPalmInwardPoints=True \
    +task.env.useForceDirectionBins=False \
    +task.env.useNormalInClustering=True \
    +task.env.stateFeatureDim=192 \
    +task.env.stateType=hash \
    +task.env.numKeyStates=32 \
    +task.env.hashLambdaBinary=1.0 \
    +task.env.stateIncludeGoal=True \
    +task.env.stateRunningMaxMode=global \
    --cfg_train=XArmAllegroHandCubeInBoxPPO \
    --seed=17 \
    --exp_name='PPO' \
    --logdir='cube_in_box' \
    --run_device_id=0 \
    --web_visualizer_port=-1
    # task.env.enableDebugVis=True \
    # task.env.visEnvNum=2 \
    # reward_type=target+bonus+success+reach+energy_reach+contact_coverage+align \
    # +task.env.visDir=cube_in_box_0_v1 \
    # reward_type=target+bonus+success+contact_coverage \

