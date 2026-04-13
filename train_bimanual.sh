#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


python src/train.py \
    num_envs=2048 \
    task=LeapBimanualArti \
    train=LeapBimanualManipulationPPO \
    headless=True \
    reward_type=task+ccge \
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
    --logdir='leap_bimanual_512_800_global_32_state_waffleiron' \
    --web_visualizer_port=-1

                    # dof_props["velocity"][i] = 100.0
                    # dof_props["effort"][i] = 10.0
                    # dof_props["stiffness"][i] = 0.1
                    # dof_props["damping"][i] = 0.02
