#!/usr/bin/env python3
from isaacgym.torch_utils import quat_apply, normalize

import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from nav_msgs.msg import Odometry
from quadruped_model.scripts.play import initialize_runner

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist

from ament_index_python.packages import get_package_share_directory

import yaml
from yaml.loader import SafeLoader
import torch
import numpy as np

import matplotlib.pyplot as plt


# @ torch.jit.script

def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


class ObservationNode(Node):
    def __init__(self, task="a1"):

        super().__init__('observation_node')
        
        self.visualize_height = True
        self.device = 'cpu'

        install_dir          = get_package_share_directory('quadruped_model')
        a1_config_path       = os.path.join(install_dir, "config", "a1_params.yaml")
        a1_rough_config_path = os.path.join(install_dir, "config", "a1_rough_params.yaml")
        aselsan_config_path  = os.path.join(install_dir, "config", "aselsan_params.yaml")
        go1_config_path      = os.path.join(install_dir, "config", "go1_params.yaml")
        cheetah_config_path  = os.path.join(install_dir, "config", "cheetah_params.yaml")

        self.joint_pos_err_last_last = torch.zeros((1, 12), device=self.device)
        self.joint_pos_err_last      = torch.zeros((1, 12), device=self.device)

        self.joint_vel_last_last = torch.zeros((1, 12), device=self.device)
        self.joint_vel_last      = torch.zeros((1, 12), device=self.device)

        if task == "a1_flat":
            print("config path: ", a1_config_path)
            config_path = a1_config_path

        elif task == "go1":
            print("config path: ", go1_config_path)
            config_path = go1_config_path

        elif task == "cheetah":
            print("config path: ", cheetah_config_path)
            config_path = cheetah_config_path

        elif task == "a1_rough":
            print("config path: ", a1_rough_config_path)
            config_path = a1_rough_config_path

        elif task == "aselsan_flat":
            print("config path: ", aselsan_config_path)
            config_path = aselsan_config_path


        self.params =  {}

        with open(config_path) as f:
            self.params = yaml.load(f, Loader=SafeLoader)

        self.joint_state = JointState()
        self.odometry = Odometry()

        self.logging = False
        self.controller = "joint_group_effort_controller"
        self.joint_trajectory_topic = "/{}/joint_trajectory".format(self.controller)
        self.joint_torque_topic = "/{}/commands".format(self.controller)

        self.join_state_pose_tensor = torch.zeros((12), device=self.device)
        self.join_state_vel_tensor  = torch.zeros((12), device=self.device)


        self.clip_obs = self.params["clip_obs"]
        self.clip_actions = self.params["clip_actions"]
        

        self.torque_limits = torch.tensor([self.params["torque_limits"]], device=self.device)

        self.damping = self.params["damping"]
        self.stiffness = self.params["stiffness"]
        self.d_gains = torch.tensor([self.damping] * 12, device=self.device)
        self.p_gains = torch.tensor([self.stiffness] * 12, device=self.device)
        self.action_scale = self.params["action_scale"]
        
        self.joint_torque_publisher = self.create_publisher(Float64MultiArray, self.joint_torque_topic, 10)

        self.odometry_subscriber = self.create_subscription(Odometry, 'odom/ground_truth', self.odometry_callback, 10)
        self.joint_state_subscriber = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.cmd_vel_subscriber = self.create_subscription(Twist, 'cmd_vel', self.cmd_vel_callback, 10)

        self.num_of_dofs = 12

        

        self.base_lin_vel = torch.tensor([] * 3, device=self.device)
        self.base_ang_vel = torch.tensor([] * 3, device=self.device)
        self.dof_pos      = torch.tensor([[] * self.num_of_dofs], device=self.device)
        self.dof_vel      = torch.tensor([[] * self.num_of_dofs], device=self.device)

        self.default_dof_pos =  torch.tensor([self.params["default_dof_pos"]], device=self.device)
        
        self.dof_vel      = torch.tensor([[0.0]*12], device=self.device)
        self.dof_pos      = self.default_dof_pos

        self.current_command = [0.0, 0.0, 0.0]

        self.mit_actions           = torch.tensor([[0.0] * self.num_of_dofs], device=self.device)
        self.commands          = torch.tensor([self.current_command], device=self.device)
        self.base_quat         = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.gravity_vec       = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
        self.projected_gravity = torch.tensor([[0.0, 0.0, -9.81]], device=self.device)  

        self.mit_commands      = torch.zeros(1, 15, device=self.device, requires_grad=False)
        self.mit_observations  = torch.zeros(1, 70, device=self.device, requires_grad=False)
        self.prev_actions      = torch.zeros(1, 12, device=self.device, requires_grad=False)
        self.clock_inputs      = torch.zeros(1, 4, dtype=torch.float, device=self.device, requires_grad=False)
        self.gait_indices      = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)
        self.dt                = 0.005 * 4

        self.calculate_command()

        self.obs_history = torch.zeros(1, 30*70, dtype=torch.float,
                                       device=self.device, requires_grad=False)

        actuator_path = "/home/mert/ros-pt/src/quadruped_model/models/a1_flat/unitree_go1.pt"
        self.actuator_network = torch.jit.load(actuator_path).to(self.device)

        self.joint_names = self.params["joint_names"]

        self.mit_commands_scale = torch.tensor([self.params["lin_vel"],              self.params["lin_vel"], self.params["ang_vel"],
                                                self.params["body_height_cmd"],      self.params["gait_freq_cmd"],
                                                self.params["gait_phase_cmd"],       self.params["gait_phase_cmd"],
                                                self.params["gait_phase_cmd"],       self.params["gait_phase_cmd"],
                                                self.params["footswing_height_cmd"], self.params["body_pitch_cmd"],
                                                self.params["body_roll_cmd"],        self.params["stance_width_cmd"],
                                                self.params["stance_length_cmd"],    self.params["aux_reward_cmd"]],
                                                device=self.device, requires_grad=False, )

        models_path            = os.path.join(install_dir, "models")
        adaptation_module_path = os.path.join(models_path, "a1_flat/adaptation_module_latest.jit")
        body_module_path       = os.path.join(models_path, "a1_flat/body_latest.jit")

        self.adaptation_module = torch.jit.load(adaptation_module_path)
        self.body_module       = torch.jit.load(body_module_path)
        

        print("Waiting for joint states and odometry...")
        while self.dof_pos.size() == torch.Size([0]) or self.dof_vel.size() == torch.Size([0]):
            rclpy.spin_once(self, timeout_sec=0.1)
        print("Joint states and odometry received.")

        self.create_timer(0.005, self.compute_observation)  
                                 
    def cmd_vel_callback(self, msg):
        # self.current_command = [msg.linear.x, msg.linear.y, msg.angular.z]      
        # self.commands        = torch.tensor([self.current_command], device=self.device)      
        pass    

    def joint_state_callback(self, msg):
        
        self.joint_state = msg
        names = msg.name
        poses = msg.position
        vels  = msg.velocity

        self.joint_state.velocity = [0.0] * 12

        joint_pose_dict = {}
        joint_vels_dict = {}

        for i in range(len(names)):
            joint_pose_dict[names[i]] = poses[i]
            joint_vels_dict[names[i]] = 0.0 if len(vels) == 0 else vels[i]

        for i in range(len(self.joint_names)):
            self.joint_state.position[i] = joint_pose_dict[self.joint_names[i]]
            self.joint_state.velocity[i] = joint_vels_dict[self.joint_names[i]]

        self.join_state_pose_tensor = torch.tensor(self.joint_state.position)
        self.join_state_vel_tensor  = torch.tensor(self.joint_state.velocity).view(1, 12)


    def odometry_callback(self, msg):             
        self.odometry = msg

    def calculate_command(self):

        gaits = {"pronking" : [0,   0,   0  ],
                 "trotting" : [0.5, 0,   0  ],
                 "bounding" : [0,   0.5, 0  ],
                 "pacing"   : [0,   0,   0.5]}

        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.0, 0.0, 0.0
        body_height_cmd = 0.0
        step_frequency_cmd = 1.0
        gait = torch.tensor(gaits["trotting"])
        footswing_height_cmd = 0.04
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25
        duration = 0.5
        stance_length_cmd = 0.5
        aux_reward_coef = 2.1376e-04 

        self.mit_commands[:, 0]   = x_vel_cmd
        self.mit_commands[:, 1]   = y_vel_cmd
        self.mit_commands[:, 2]   = yaw_vel_cmd
        self.mit_commands[:, 3]   = body_height_cmd
        self.mit_commands[:, 4]   = step_frequency_cmd
        self.mit_commands[:, 5:8] = gait
        self.mit_commands[:, 8]   = duration
        self.mit_commands[:, 9]   = footswing_height_cmd
        self.mit_commands[:, 10]  = pitch_cmd
        self.mit_commands[:, 11]  = roll_cmd
        self.mit_commands[:, 12]  = stance_width_cmd
        self.mit_commands[:, 13]  = stance_length_cmd
        self.mit_commands[:, 14]  = aux_reward_coef

        
        print("commands shape: ", self.mit_commands.shape)
        return self.mit_commands   

    def step_contact_targets(self):

        frequencies       = self.mit_commands[:, 4]
        phases            = self.mit_commands[:, 5]
        offsets           = self.mit_commands[:, 6]
        bounds            = self.mit_commands[:, 7]
        durations         = self.mit_commands[:, 8]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)


        foot_indices = [self.gait_indices + phases + offsets + bounds,
                        self.gait_indices + offsets,
                        self.gait_indices + bounds,
                        self.gait_indices + phases]

        self.foot_indices = torch.remainder(torch.cat([foot_indices[i].unsqueeze(1) for i in range(4)], dim=1), 1.0)

        for idxs in foot_indices:
                stance_idxs = torch.remainder(idxs, 1) < durations
                swing_idxs = torch.remainder(idxs, 1) > durations

                idxs[stance_idxs] = torch.remainder(idxs[stance_idxs], 1) * (0.5 / durations[stance_idxs])
                idxs[swing_idxs] = 0.5 + (torch.remainder(idxs[swing_idxs], 1) - durations[swing_idxs]) * (
                            0.5 / (1 - durations[swing_idxs]))

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])


    def compute_observation(self):
        
        start_time = time.time()

        self.base_quat    = torch.tensor([[self.odometry.pose.pose.orientation.x, 
                                           self.odometry.pose.pose.orientation.y, 
                                           self.odometry.pose.pose.orientation.z, 
                                           self.odometry.pose.pose.orientation.w]], 
                                           device=self.device)
        
        self.dof_pos      = torch.tensor([self.joint_state.position[:]], device=self.device)
        self.dof_vel      = torch.tensor([self.joint_state.velocity[:]], device=self.device)
        
        self.projected_gravity = self.quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.step_contact_targets()
        self.mit_observations = torch.cat((self.projected_gravity,
                                           self.mit_commands * self.mit_commands_scale,
                                           (self.dof_pos - self.default_dof_pos) * self.params["dof_pos"],
                                           self.dof_vel * self.params["dof_vel"],
                                           self.mit_actions,
                                           self.prev_actions,
                                           self.clock_inputs
                                           ), dim=-1)

        self.mit_observations = torch.clip(self.mit_observations, -self.clip_obs, self.clip_obs)

        self.obs_history = torch.cat((self.obs_history[:, 70:], self.mit_observations), dim=-1)
        self.latent = self.adaptation_module.forward(self.obs_history.cpu())
        # self.latent = torch.tensor([[1.5, -0.5]])
        print("latent: ",self.latent)


        self.mit_actions = self.body_module.forward(torch.cat((self.obs_history.cpu(), self.latent), dim=-1))
        self.actions = torch.clip(self.mit_actions, -self.clip_actions, self.clip_actions).to(self.device)
                
        self.prev_actions = self.mit_actions.to(self.device).clone().detach()

        # self.mit_actions = torch.clip(self.mit_actions, -self.clip_actions, self.clip_actions).to(self.device)


        # ------------------- Publish Joint Torques ------------------- #

        # torques = self.compute_torques(self.mit_actions * self.action_scale)
        torques = self._compute_torques(self.mit_actions)


        self.publish_trajectory_torques(torques.tolist()[0])
        
        end_time = time.time()
        duration = end_time - start_time
        # print(duration)
        

    def quat_rotate_inverse(self, q, v):
        shape = q.shape
        q_w = q[:, -1]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
        b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
        c = q_vec * \
            torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
                shape[0], 3, 1)).squeeze(-1) * 2.0
        return a - b + c


    def publish_trajectory_torques(self, torques):
        arr = Float64MultiArray()

        arr.data = torques
        print("torques: ", arr.data)
        # print(torques, "\n -------------- \n")
        self.joint_torque_publisher.publish(arr)


    # def compute_torques(self, actions):

    #     torques = self.p_gains*(actions + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

    #     return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _compute_torques(self, actions):

        actions_scale = 0.25
        hip_scale_reduction = 0.5

        actions_scaled = actions[:, :12] * actions_scale
        actions_scaled[:, [0, 3, 6, 9]] *=  hip_scale_reduction  # scale down hip flexion range

        self.joint_pos_target = actions_scaled + self.default_dof_pos
        

        self.joint_pos_err = self.join_state_pose_tensor - self.joint_pos_target
        self.joint_vel     = self.join_state_vel_tensor

        controller = "P"
        # controller = "actuator_network"

        if controller == "actuator_network":

            torques = self.eval_actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel.view(1, 12), self.joint_vel_last.view(1, 12), self.joint_vel_last_last.view(1, 12))
            
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last      = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last     = torch.clone(self.joint_vel_last)
            self.joint_vel_last          = torch.clone(self.joint_vel)

        elif controller == "P":

            torques = 20 * 1 * (
                        self.joint_pos_target - self.dof_pos + 0) - 0.5 * 1 * self.dof_vel


        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def eval_actuator_network(self, joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = self.actuator_network(xs.view(1 * 12, 6))
                return torques.view(1, 12)
    


if __name__ == '__main__':
    rclpy.init()


    if len(sys.argv) > 1:
        task_name = str(sys.argv[1])
    else:
        task_name = "a1"

    observation_node = ObservationNode(task_name)

    rclpy.spin(observation_node)
    observation_node.destroy_node()
    rclpy.shutdown()

