import os


import isaacgym
from hydra._internal.utils import get_args_parser
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

from algorithms.ppo import PPO
from tasks import load_isaacgym_env
from utils.config import get_args, load_cfg

# from utils.vis import Visualizer # use visualizer requires to install sim-web-visualizer

def list_available_checkpoints(model_dir):
    import glob
    import re
    
    if not os.path.isdir(model_dir):
        return [], []
    
    checkpoint_pattern = os.path.join(model_dir, "model_*.pt")
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    iterations = []
    for file_path in checkpoint_files:
        filename = os.path.basename(file_path)
        match = re.match(r"model_(\d+)\.pt", filename)
        if match:
            iterations.append(int(match.group(1)))
            
    checkpoints_path = [os.path.join(model_dir, f"model_{iter}.pt") for iter in sorted(iterations)]
    
    return checkpoints_path, sorted(iterations)

if __name__ == "__main__":
    set_np_formatting()

    # argparse
    parser = get_args_parser()
    parser.add_argument(
        "--num_iterations", type=int, default=1000, help="Number of iterations to run"
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed Number")
    parser.add_argument("--run_device_id", type=int, default=0, help="Device id")

    parser.add_argument("--torch_deterministic", action="store_true", default=False, help="Apply additional PyTorch settings for more deterministic behaviour")
    parser.add_argument("--test", action="store_true", default=False, help="Run trained policy, no training",)
    parser.add_argument("--con", action="store_true", default=False, help="whether continue train")
    parser.add_argument("--web_visualizer_port", type=int, default=-1, help="port to visualize in web visualizer, set to -1 to disable")
    parser.add_argument("--collect_demo_num", type=int, default=-1, help="collect demo num")
    parser.add_argument("--eval_times", type=int, default=5, help="Eval times for each object")
    parser.add_argument("--max_iterations", type=int, default=-1, help="Max iterations for training")
    parser.add_argument("--resume_iter", type=int, default=None, help="Resume from specific iteration (default: latest)")
    parser.add_argument("--list_checkpoints", action="store_true", default=False, help="List available checkpoints and exit")

    parser.add_argument("--cfg_train",type=str,default="XArmAllegroHandFunctionalManipulationUnderarmPPO",help="Training config")

    parser.add_argument("--logdir", type=str, default="", help="Log directory")
    parser.add_argument("--print_log", type=lambda x: x.lower() == "true", default=True, help="Print log (True/False)")
    parser.add_argument("--method", type=str, default="", help="Method name")
    parser.add_argument("--exp_name", type=str, default="", help="Exp name")
    parser.add_argument("--model_dir", type=str, default="", help="Choose a model dir")
    parser.add_argument("--eval_name", type=str, default="", help="Eval metric saving name")
    parser.add_argument("--vis_env_num", type=int, default=0, help="Number of env to visualize")

    # score matching parameter
    parser.add_argument("--t0", type=float, default=0.05, help="t0 for sample")
    parser.add_argument("--hidden_dim", type=int, default=1024, help="num of hidden dim")
    parser.add_argument("--embed_dim", type=int, default=512, help="num of embed_dim")
    parser.add_argument("--score_mode", type=str, default="target", help="score mode")
    parser.add_argument("--space", type=str, default="euler", help="angle space")
    parser.add_argument("--cond_on_arm", action="store_true", help="dual score")
    parser.add_argument("--n_obs_steps", type=int, default=2, help="observation steps")
    parser.add_argument("--n_action_steps", type=int, default=1)
    parser.add_argument("--n_prediction_steps", type=int, default=4)
    parser.add_argument("--encode_state_type", type=str, default="all", help="encode state type")
    parser.add_argument("--score_action_type", type=str, default="all", metavar="SCORE_ACTION_TYPE", help="score action type: arm, hand, all")
    parser.add_argument("--action_mode", type=str, default="rel", metavar="ACTION_MODE", help="action mode: rel, abs, obs", )
    parser.add_argument("--score_model_path",  type=str, default="/home/thwu/Projects/func-mani/ckpt/score_all.pt", help="pretrain score model path")

    parser.add_argument("--use_curiosity_model", action="store_true", default=False, help="Whether to use curiosity model")
    parser.add_argument("--model_type", type=str, default="prediction_error", help="Curiosity model type: prediction_error, rnd, disagreement, neural_hash")
    parser.add_argument("--intrinsic_reward_scale", type=float, default=1.0, help="Scale for intrinsic reward from curiosity model")

    args = parser.parse_args()

    sim_device = f"cuda:{args.run_device_id}" if args.run_device_id >= 0 else "cpu"
    rl_device = f"cuda:{args.run_device_id}" if args.run_device_id >= 0 else "cpu"

    cfg_train, logdir = load_cfg(args)

    # set the seed for reproducibility
    set_seed(args.seed)
    """Change for different methods."""



    if args.exp_name == "PPO":

        leap_bimanual_flags = [
            "task=LeapBimanualArti",
            "task=LeapBimanualBoardLift",
        ]
        if "task=XArmAllegroHandFunctionalManipulationUnderarm" in args.overrides \
            or "task=XArmAllegroHandCubeInBox" in args.overrides \
            or "task=XArmAllegroHandTableTop" in args.overrides \
            or "task=XArmLeapHandFunctionalManipulationUnderarm" in args.overrides \
            or "task=XArmLeapHandCubeInBox" in args.overrides \
            or "task=XArmLeapHandTableTop" in args.overrides \
            or "task=XArm7LeapHandFunctionalManipulationUnderarm" in args.overrides \
            or "task=XArm7LeapHandCubeInBox" in args.overrides:
            obs_space = [
                # Hand state
                "xarm_endeffector_position",
                "xarm_endeffector_orientation",
                "xarm_endeffector_linear_velocity",
                "xarm_endeffector_angular_velocity",
                "allegro_hand_dof_position",
                "allegro_hand_dof_velocity",
                # Fingertip information for contact detection
                "fingertip_position_wrt_palm",
                "fingertip_orientation_wrt_palm",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                
                # extra obs
                "fingertip_position_wrt_object",
                "fingertip_orientation_wrt_object",
                "fingertip_geometric_distance",
                "fingertip_geometric_direction",
                
                # Object state (current target object)
                "object_position_wrt_palm",
                "object_orientation_wrt_palm",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                "nearest_non_target_object_position",
                "nearest_non_target_object_orientation",
                # Scene understanding
                "object_bbox",
                # Contact information
                "tactile", # 0.1N-binary signal 
                # Action
                "action",
            ]
            action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]

        elif any(flag in args.overrides for flag in leap_bimanual_flags):
            if "task=LeapBimanualBoardLift" in args.overrides:
                obs_space = [
                    "rh_root_position",
                    "rh_root_orientation",
                    "rh_root_linear_velocity",
                    "rh_root_angular_velocity",
                    "lh_root_position",
                    "lh_root_orientation",
                    "lh_root_linear_velocity",
                    "lh_root_angular_velocity",
                    "rh_dof_position",
                    "rh_dof_velocity",
                    "lh_dof_position",
                    "lh_dof_velocity",
                    "object_position",
                    "object_orientation",
                    "object_linear_velocity",
                    "object_angular_velocity",
                    "object_position_wrt_rh_palm",
                    "object_orientation_wrt_rh_palm",
                    "object_position_wrt_lh_palm",
                    "object_orientation_wrt_lh_palm",
                    "hand_relative_position",
                    "hand_relative_orientation",
                    "rh_fingertip_position",
                    "rh_fingertip_orientation",
                    "rh_fingertip_linear_velocity",
                    "rh_fingertip_angular_velocity",
                    "lh_fingertip_position",
                    "lh_fingertip_orientation",
                    "lh_fingertip_linear_velocity",
                    "lh_fingertip_angular_velocity",
                    "tactile",
                    "action",
                ]
            else:
                obs_space = [
                    # Hand roots
                    "rh_root_position",
                    "rh_root_orientation",
                    "rh_root_linear_velocity",
                    "rh_root_angular_velocity",
                    "lh_root_position",
                    "lh_root_orientation",
                    "lh_root_linear_velocity",
                    "lh_root_angular_velocity",
                    # Hand DOFs
                    "rh_dof_position",
                    "rh_dof_velocity",
                    "lh_dof_position",
                    "lh_dof_velocity",
                    # Object (world)
                    "object_position",
                    "object_orientation",
                    "object_linear_velocity",
                    "object_angular_velocity",
                    # Articulated object parts (top & bottom)
                    "top_part_position",
                    "top_part_orientation",
                    "top_part_linear_velocity",
                    "top_part_angular_velocity",
                    "bottom_part_position",
                    "bottom_part_orientation",
                    "bottom_part_linear_velocity",
                    "bottom_part_angular_velocity",
                    "object_dof_position",
                    "object_dof_velocity",
                    "object_position_wrt_rh_palm",
                    "object_orientation_wrt_rh_palm",
                    "object_position_wrt_lh_palm",
                    "object_orientation_wrt_lh_palm",
                    "hand_relative_position",
                    "hand_relative_orientation",
                    "rh_fingertip_position",
                    "rh_fingertip_orientation",
                    "rh_fingertip_linear_velocity",
                    "rh_fingertip_angular_velocity",
                    "lh_fingertip_position",
                    "lh_fingertip_orientation",
                    "lh_fingertip_linear_velocity",
                    "lh_fingertip_angular_velocity",
                    "tactile",
                    "action",
                ]
            action_space = [
                "rh_root_translation",
                "lh_root_translation",
                # "rh_root_rotation_6d",
                # "lh_root_rotation_6d",
                # "rh_eef_rotation_quaternion",
                # "lh_eef_rotation_quaternion",
                "rh_wrist_rotation",
                "lh_wrist_rotation",
                "lh_finger_positions",
                "rh_finger_positions",
            ]
        elif "task=InhandManipulationAllegro" in args.overrides \
            or "task=InhandManipulationLeap" in args.overrides:
            obs_space = [
                # Hand state
                "allegro_hand_dof_position",
                "allegro_hand_dof_velocity",
                "allegro_hand_dof_force",
                # Fingertip information for contact detection
                # "fingertip_position_wrt_palm",
                # "fingertip_orientation_wrt_palm",
                # "fingertip_linear_velocity",
                # "fingertip_angular_velocity",
                # Object state (current target object)
                # "object_position_wrt_palm",
                # "object_orientation_wrt_palm",
                "goal_position",
                "goal_orientation",
                "goal_orientation_dist",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                # Action
                "action",
            ]
            action_space = ["hand_rotation"]
        elif "task=InhandManipulationShadow" in args.overrides:
            obs_space = [
                # Hand state
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "shadow_hand_dof_force",
                # Fingertip information for contact detection
                # "fingertip_position_wrt_palm",
                # "fingertip_orientation_wrt_palm",
                # "fingertip_linear_velocity",
                # "fingertip_angular_velocity",
                # Object state (current target object)
                # "object_position_wrt_palm",
                # "object_orientation_wrt_palm",
                "goal_position",
                "goal_orientation",
                "goal_orientation_dist",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                # Action
                "action",
            ]
            action_space = ["hand_rotation"]
        elif "task=InhandManipulationShadowSpin" in args.overrides:
            obs_space = [
                # Hand state
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "shadow_hand_dof_force",
                # Fingertip information for contact detection
                # "fingertip_position_wrt_palm",
                # "fingertip_orientation_wrt_palm",
                # "fingertip_linear_velocity",
                # "fingertip_angular_velocity",
                # Object state (current target object)
                # "object_position_wrt_palm",
                # "object_orientation_wrt_palm",
                "goal_position",
                "goal_orientation",
                "goal_orientation_dist",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                # Action
                "action",
            ]
            action_space = ["hand_rotation"]
        elif "task=InhandManipulationShadowSpinUpsideDown" in args.overrides:
            obs_space = [
                # Hand state
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "shadow_hand_dof_force",
                # Fingertip information for contact detection
                # "fingertip_position_wrt_palm",
                # "fingertip_orientation_wrt_palm",
                # "fingertip_linear_velocity",
                # "fingertip_angular_velocity",
                # Object state (current target object)
                # "object_position_wrt_palm",
                # "object_orientation_wrt_palm",
                "goal_position",
                "goal_orientation",
                "goal_orientation_dist",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                # Action
                "action",
            ]
            action_space = ["hand_rotation"]
        elif "task=PushBox2DPoint" in args.overrides:
            obs_space = [
                "point_xy_obs"
            ]
            action_space = ["point_xy_acts"]
        else:
            # Default Shadow Hand configuration
            obs_space = [
                "ur_endeffector_position",
                "ur_endeffector_orientation",
                "shadow_hand_dof_position",
                "shadow_hand_dof_velocity",
                "fingertip_position_wrt_palm",
                "fingertip_orientation_wrt_palm",
                "fingertip_linear_velocity",
                "fingertip_angular_velocity",
                "object_position_wrt_palm",
                "object_orientation_wrt_palm",
                "object_position",
                "object_orientation",
                "object_linear_velocity",
                "object_angular_velocity",
                "object_target_relposecontact",
                "position_error",
                "orientation_error",
                "fingerjoint_error",
                "object_bbox",
            ]
            action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]

        # training parameter
        cfg_train["learn"]["nsteps"] = 16
        cfg_train["learn"]["noptepochs"] = 5
        cfg_train["learn"]["nminibatches"] = 4
        cfg_train["learn"]["desired_kl"] = 0.016
        cfg_train["learn"]["gamma"] = 0.99
    elif args.exp_name == "ppo_real":
        # Check if using XArm Allegro task or Shadow Hand task
        if "task=XArmAllegroHandFunctionalManipulationUnderarm" in args.overrides:
            obs_space = [
                "xarm_endeffector_position",
                "xarm_endeffector_orientation",
                "allegro_hand_dof_position",
                "object_position_wrt_palm",
                "object_orientation_wrt_palm",
                "object_target_relposecontact",
            ]
        else:
            obs_space = [
                "ur_endeffector_position",
                "ur_endeffector_orientation",
                "shadow_hand_dof_position",
                "object_position_wrt_palm",
                "object_orientation_wrt_palm",
                "object_target_relposecontact",
            ]
        action_space = ["wrist_translation", "wrist_rotation", "hand_rotation"]
        # training parameter
        cfg_train["learn"]["nsteps"] = 8
        cfg_train["learn"]["noptepochs"] = 5
        cfg_train["learn"]["nminibatches"] = 4
        cfg_train["learn"]["desired_kl"] = 0.016
        cfg_train["learn"]["gamma"] = 0.99
    else:
        raise NotImplementedError(f"setting {args.exp_name} not supported")
    """
    load env
    """
    # override env args
    args.overrides.append(f"seed={args.seed}")
    args.overrides.append(f"sim_device={sim_device}")
    args.overrides.append(f"rl_device={rl_device}")
    args.overrides.append(f"obs_space={obs_space}")
    args.overrides.append(f"action_space={action_space}")

    cfg_train["setting"]["use_curiosity_model"] = args.use_curiosity_model
    cfg_train["curiosity_model"]["model_type"] = args.model_type
    cfg_train["learn"]["intrinsic_reward_scale"] = args.intrinsic_reward_scale
    # Load and wrap the Isaac Gym environment
    env, _ = load_isaacgym_env(
        task_name="", args=args
    )
    """
    load agent
    """
    learn_cfg = cfg_train["learn"]
    if "mode=eval" in args.overrides:
        learn_cfg["test"] = True
    is_testing = learn_cfg["test"]
    # Override resume and testing flags if they are passed as parameters.
    if args.con or learn_cfg["test"]:
        if args.model_dir is None or args.model_dir == "":
            raise ValueError("model_dir is required when con is True")
        checkpoints_path, iterations = list_available_checkpoints(args.model_dir)
        if len(checkpoints_path) == 0:
            raise ValueError("No checkpoints found in model_dir")
        if args.resume_iter is not None:
            chkpt_path = checkpoints_path[iterations.index(args.resume_iter)]
        else:
            chkpt_path = checkpoints_path[-1]
        print(f"Loading checkpoint from {chkpt_path} at iteration {args.resume_iter if args.resume_iter is not None else iterations[-1]}")
        # breakpoint()

    runner = PPO(
        vec_env=env,
        cfg_train=cfg_train,
        device=rl_device,
        sampler=learn_cfg.get("sampler", "sequential"),
        log_dir=logdir,
        is_testing=is_testing,
        print_log=learn_cfg["print_log"],
        apply_reset=False,
        asymmetric=False,
        args=args,
    )

    iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations

    if args.model_dir != "":
        if is_testing:
            runner.restore_test(chkpt_path)
            runner.eval(0)
        else:
            runner.restore_train(chkpt_path)
            runner.train(
                num_learning_iterations=iterations,
                log_interval=cfg_train["learn"]["save_interval"],
            )

    else:  # train from scratch
        runner.train(
            num_learning_iterations=iterations,
            log_interval=cfg_train["learn"]["save_interval"],
        )
