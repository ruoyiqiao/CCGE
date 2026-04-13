#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python src/train.py \
    num_envs=2048 \
    task=LeapBimanualBoardLift \
    train=LeapBimanualManipulationPPO \
    headless=True \
    reward_type=targ+ccge \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    physics_engine=physx \
    +task.env.stateFeatureDim=96 \
    +task.env.stateType=hash \
    +task.env.numKeyStates=32 \
    +task.env.hashNoiseScale=0.3 \
    +task.env.hashLambdaBinary=1.0 \
    +task.env.stateIncludeGoal=False \
    +task.env.stateRunningMaxMode=global \
    --cfg_train=LeapBimanualManipulationPPO \
    --seed=0 \
    --exp_name=PPO \
    --run_device_id=0 \
    --logdir='bimanual_board_lift' \
    --web_visualizer_port=-1

# Optional pose overrides for both arm bases (Hydra CLI examples):
# +task.env.handBasePoseOverrides.right.position='[-0.24,0.24,0.55]' \
# +task.env.handBasePoseOverrides.left.position='[0.24,0.24,0.55]' \
# +task.env.handBasePoseOverrides.right.orientation='[0.0,0.0,0.0,1.0]' \
# +task.env.handBasePoseOverrides.left.orientation='[0.0,0.0,0.0,1.0]' \
