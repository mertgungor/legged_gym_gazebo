import sys
import os
rsl_rl_path = os.path.join(os.path.expanduser('~'), "rsl_rl/rsl_rl")
sys.path.append(rsl_rl_path)

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from typing import Tuple, Union
import torch


class Env(VecEnv):
    def __init__(self, 
                num_envs = 1, 
                num_obs = 48, 
                num_privileged_obs = 0, 
                num_actions = 12, 
                max_episode_length = 1000.0, 
                ):
        super()
        device = 'cuda'
        self.num_envs           = num_envs
        self.num_obs            = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.num_actions        = num_actions
        self.max_episode_length = max_episode_length
        self.privileged_obs_buf = None
        self.obs_buf            = torch.zeros(num_envs, num_obs, dtype=torch.float32, device=device)
        self.rew_buf            = torch.tensor([0.], device=device)
        self.reset_buf          = torch.tensor([1], device=device) # 1 if reset, 0 otherwise
        self.episode_length_buf = torch.tensor([0], device=device) # current episode duration
        self.extras             = {}
        self.device             = device
        

    def get_observations(self) -> torch.Tensor:
        return self.obs_buf

    def get_privileged_observations(self) -> torch.Tensor:
        return self.privileged_obs_buf

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, Union[torch.Tensor, None], torch.Tensor, torch.Tensor, dict]:
        pass
    
    def reset(self, env_ids=[1]):
        return [], []


def initialize_runner(model_path, params):

    current_dir = os.path.dirname(os.path.realpath(__file__))
    
    log_dir = os.path.join(current_dir, "../../logs")

    print("model_path: ", model_path)
    print("log_dir: ", log_dir)

    device_type = 'cuda'
    print("device_type: ", device_type)

    num_flat_features = params["num_observations"]

    env = Env(num_obs=params["num_observations"])

    env.num_envs           = 1 
    env.num_obs            = num_flat_features 
    env.num_privileged_obs = None
    env.num_actions        = 12 
    env.max_episode_length = 1001.0
    env.privileged_obs_buf = None
    env.obs_buf            = torch.zeros(1, params["num_observations"], dtype=torch.float32, device=device_type)
    env.rew_buf            = torch.tensor([0.], device=device_type)
    env.reset_buf          = torch.tensor([1], device=device_type) 
    env.episode_length_buf = torch.tensor([0], device=device_type)  # current episode duration
    env.extras             = {} 
    env.device             = device_type

    train_cfg_dict = {
                    'algorithm': 
                        {
                        'clip_param'            : params["algorithm"]["clip_param"],
                        'desired_kl'            : params["algorithm"]["desired_kl"],
                        'entropy_coef'          : params["algorithm"]["entropy_coef"],
                        'gamma'                 : params["algorithm"]["gamma"],
                        'lam'                   : params["algorithm"]["lam"],
                        'learning_rate'         : params["algorithm"]["learning_rate"],
                        'max_grad_norm'         : params["algorithm"]["max_grad_norm"],
                        'num_learning_epochs'   : params["algorithm"]["num_learning_epochs"],
                        'num_mini_batches'      : params["algorithm"]["num_mini_batches"],
                        'schedule'              : params["algorithm"]["schedule"],
                        'use_clipped_value_loss': params["algorithm"]["use_clipped_value_loss"],
                        'value_loss_coef'       : params["algorithm"]["value_loss_coef"]
                        },
                    'init_member_classes': {},
                    'policy': 
                        {
                        'activation'        :  params["policy"]["activation"],
                        'actor_hidden_dims' :  params["policy"]["actor_hidden_dims"],
                        'critic_hidden_dims':  params["policy"]["critic_hidden_dims"],
                        'init_noise_std'    :  params["policy"]["init_noise_std"],
                        },
                    'runner': 
                        {
                        'algorithm_class_name': params["runner"]["algorithm_class_name"],
                        'checkpoint'          : params["runner"]["checkpoint"],
                        'experiment_name'     : params["runner"]["experiment_name"],
                        'load_run'            : params["runner"]["load_run"],
                        'max_iterations'      : params["runner"]["max_iterations"],
                        'num_steps_per_env'   : params["runner"]["num_steps_per_env"],
                        'policy_class_name'   : params["runner"]["policy_class_name"], 
                        'resume'              : params["runner"]["resume"],
                        'resume_path'         : params["runner"]["resume_path"],
                        'run_name'            : params["runner"]["run_name"],
                        'save_interval'       : params["runner"]["save_interval"],
                        } 
                    }

    runner = OnPolicyRunner(env, train_cfg_dict, log_dir, device=device_type)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=env.device)

    return policy
    

