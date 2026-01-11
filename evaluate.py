import argparse
import torch
import genesis as gs
from go2_env import Go2Env
from agent import Agent

import os

ag_cfg = Agent.command_cfg

def get_eval_cfgs():
    """
    Returns the standardized configuration for evaluation.
    This ensures all agents are evaluated on the exact same task.
    """
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0, "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8, "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5, "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": ag_cfg["kp"],
        "kd": ag_cfg["kd"],
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 9999.0, # Disable resampling to keep command fixed
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 45,
        "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
    }
    reward_cfg = {
        "tracking_sigma": 0.25, "base_height_target": 0.3, "feet_height_target": 0.075,
        "reward_scales": {}, # Rewards don't matter for evaluation, only the metric
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": ag_cfg["lin_vel_x_range"],
        "lin_vel_y_range": ag_cfg["lin_vel_y_range"],
        "ang_vel_range": ag_cfg["ang_vel_range"]
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--visualize", action="store_true", help="Show the viewer")
    parser.add_argument("-n", "--n_runs", type=int, default=5, help="Number of runs")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    # Load Standardized Configs
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_eval_cfgs()

    # Create Environment
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.visualize,
    )

    # Initialize User Agent
    try:
        agent = Agent()
        agent.reset()
    except Exception as e:
        print(f"Error: Failed to initialize Agent: {e}")
        return

    displacements = []

    for _ in range(args.n_runs):
        # Evaluation Loop
        obs, _ = env.reset()
        
        start_pos = env.base_pos[0, :2].clone()

        total_valid_distance = 0.0
        dt = env.dt
        max_steps = int(env_cfg["episode_length_s"] / dt)
        
        terminate_flag = False
        with torch.no_grad():
            for step in range(max_steps):
                # 1. Get Action from Agent
                actions = agent.apply(obs)

                # 2. Step Environment
                obs, _, dones, _ = env.step(actions)

                # 3. Calculate Metric
                # Get current velocity (xy plane)
                lin_vel_xy = env.base_lin_vel[:, :2]
                velocity_mag = torch.norm(lin_vel_xy, dim=1).item()

                # Check validity condition: 0.5 <= velocity <= 1.0
                if 0.5 <= velocity_mag <= 1.0:
                    distance_step = velocity_mag * dt
                    total_valid_distance += distance_step

                # Handle early termination (falling over)
                if dones.any():
                    terminate_flag = True
                    break

        end_pos = env.base_pos[0, :2]
        displacement = torch.norm(end_pos - start_pos).item()

        if terminate_flag:
            displacements.append(0.0)
        else:
            displacements.append(displacement)

    score = sum(displacements) / args.n_runs
    print(score)
    pth = os.path.join(os.getcwd(), "scorecard.txt")
    with open(pth, 'w') as fp:
        fp.write(str(score))

if __name__ == "__main__":
    main()
