#!/bin/bash

# Training script for XArmAllegroHandFunctionalManipulationUnderarm Singulation & Grasp Task
# This script runs PPO training for the singulation and grasp task

python src/train.py \
    headless=True \
    env_info=True \
    num_envs=2048 \
    num_objects=5 \
    num_objects_per_env=1 \
    graphics_device_id=0 \
    split='train' \
    cluster=0 \
    task=InhandManipulationLeap \
    train=InhandManipulationLeapPPO \
    reward_type=task+reach+energy_reach+contact_coverage \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    physics_engine=physx \
    task.env.objectMode=urdf \
    task.env.enableRenderedPointCloud=False \
    task.env.enableRenderedPointCloudTargetMask=False \
    task.env.objectUrdfPath=contactdb/elephant/elephant.urdf \
    +task.env.use_center_collision=False \
    +task.env.maskBackfacePoints=True \
    +task.env.maskPalmInwardPoints=True \
    +task.env.useForceDirectionBins=False \
    +task.env.useNormalInClustering=True \
    ++task.env.enableDebugVis=False \
    +task.env.useUpsideDown=False \
    +task.env.stateFeatureDim=192 \
    +task.env.stateType=hash \
    +task.env.numKeyStates=32 \
    +task.env.hashLambdaBinary=1.0 \
    +task.env.stateIncludeGoal=True \
    +task.env.stateRunningMaxMode=global \
    --cfg_train=InhandManipulationLeapPPO \
    --seed=42 \
    --exp_name='PPO' \
    --logdir='leap_inhand_ppo' \
    --run_device_id=0 \
    --web_visualizer_port=-1 \
    # --con \
