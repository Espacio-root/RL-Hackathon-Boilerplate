import torch
from rsl_rl.modules import ActorCritic
import pickle

import os

class Agent:
    command_cfg = {
        "kp":20.0,
        "kd":0.5,
        "lin_vel_x_range": [0.6, 0.6],
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }

    def __init__(self):
        """
        Initialize the agent and load the model.
        Users can modify the path to load their specific submitted model.
        """
        # Path to the model checkpoint (submitted by user)
        self.model_path = "checkpoints/model_1500.pt"
        self.model_path = os.path.join(os.getcwd(), self.model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        config_path = "checkpoints/cfgs.pkl"
        config_path = os.path.join(os.getcwd(), config_path)
        with open(config_path, 'rb') as f:
            cfg = pickle.load(f)

        n_obs = cfg[1]['num_obs']
        policy_cfg = cfg[4]['policy'] 
        self.policy = ActorCritic(
            num_actor_obs=n_obs,
            num_critic_obs=n_obs,
            activation=policy_cfg['activation'],
            num_actions=cfg[0]['num_actions'],
            actor_hidden_dims=policy_cfg['actor_hidden_dims'],
            critic_hidden_dims=policy_cfg['critic_hidden_dims'],
            init_noise_std=policy_cfg["init_noise_std"],
        ).to(self.device)

        # Load weights
        try:
            print(f"Loading model from {self.model_path}...")
            loaded_dict = torch.load(self.model_path, map_location=self.device)
            
            # Handle different saving formats (rsl_rl saves full state dict)
            if 'model_state_dict' in loaded_dict:
                self.policy.load_state_dict(loaded_dict['model_state_dict'])
            else:
                self.policy.load_state_dict(loaded_dict)
                
            self.policy.eval()
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Warning: {self.model_path} not found. Agent will use random weights.")

    def reset(self):
        """
        Reset internal state if necessary (e.g., for RNNs).
        Stateless MLPs generally do not need this.
        """
        pass

    def apply(self, obs):
        """
        Get the action from the agent.
        
        Args:
            obs (torch.Tensor): Observation tensor of shape (num_envs, num_obs)
            
        Returns:
            torch.Tensor: Action tensor of shape (str(score))
        """
        with torch.no_grad():
            # The default ActorCritic returns (actions, log_probs, ...)
            # We only need the actions for inference
            actions = self.policy.act_inference(obs)
        return actions

if __name__ == "__main__":
    # Simple test to ensure the agent loads
    agent = Agent()
    print("Agent initialized successfully.")
