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
    task=PushBox2DPoint \
    train=PushBox2DPointPPO \
    reward_type=task+reach+energy_reach+contact_coverage \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    physics_engine=physx \
    +task.env.use_center_collision=True \
    +task.env.maskBackfacePoints=True \
    +task.env.maskPalmInwardPoints=False \
    +task.env.useForceDirectionBins=False \
    +task.env.useNormalInClustering=True \
    ++task.env.enableDebugVis=False \
    +task.env.useUpsideDown=False \
    +task.env.stateFeatureDim=192 \
    +task.env.stateType=hash \
    +task.env.numKeyStates=2 \
    +task.env.hashNoiseScale=0.3 \
    +task.env.hashLambdaBinary=0.0 \
    +task.env.randomInit=True \
    +task.env.stateIncludeGoal=True \
    +task.env.stateRunningMaxMode=global \
    --cfg_train=PushBox2DPointPPO \
    --seed=420 \
    --exp_name='PPO' \
    --logdir='push_box_2d_point_ppo' \
    --run_device_id=-1 \
    --web_visualizer_port=-1 \
    # --con \
