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
        self.device = 'cuda:0'

        install_dir          = get_package_share_directory('quadruped_model')
        a1_config_path       = os.path.join(install_dir, "config", "a1_params.yaml")
        a1_rough_config_path = os.path.join(install_dir, "config", "a1_rough_params.yaml")
        aselsan_config_path  = os.path.join(install_dir, "config", "aselsan_params.yaml")
        go1_config_path      = os.path.join(install_dir, "config", "go1_params.yaml")
        cheetah_config_path  = os.path.join(install_dir, "config", "cheetah_params.yaml")


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

        self.logging = False
        self.controller = "joint_group_effort_controller"
        self.joint_trajectory_topic = "/{}/joint_trajectory".format(self.controller)
        self.joint_torque_topic = "/{}/commands".format(self.controller)


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

        self.joint_state = JointState()
        self.odometry = Odometry()

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


        self.obs_buf = torch.tensor([], device=self.device)
        self.obs_history = torch.zeros(1, 30*70, dtype=torch.float,
                                       device=self.device, requires_grad=False)


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


    def odometry_callback(self, msg):             
        self.odometry = msg

    def calculate_command(self):

        gaits = {"pronking": [0,   0,   0  ],
                "trotting" : [0.5, 0,   0  ],
                "bounding" : [0,   0.5, 0  ],
                "pacing"   : [0,   0,   0.5]}

        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
        body_height_cmd = 0.0
        step_frequency_cmd = 3.0
        gait = torch.tensor(gaits["trotting"])
        footswing_height_cmd = 0.08
        pitch_cmd = 0.0
        roll_cmd = 0.0
        stance_width_cmd = 0.25
        duration = 0.5

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
        self.mit_commands[:, 13]  = 4.2803e-01 
        self.mit_commands[:, 14]  = 2.1376e-04 

        
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

        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * foot_indices[3])


    def compute_observation(self):
        
        start_time = time.time()

        base_lin_vel_list = torch.tensor([[self.odometry.twist.twist.linear.x, self.odometry.twist.twist.linear.y, self.odometry.twist.twist.linear.z   ]], device=self.device)
        base_ang_vel_list = torch.tensor([[self.odometry.twist.twist.angular.x, self.odometry.twist.twist.angular.y, self.odometry.twist.twist.angular.z]], device=self.device)

        self.base_quat    = torch.tensor([[self.odometry.pose.pose.orientation.x, self.odometry.pose.pose.orientation.y, self.odometry.pose.pose.orientation.z, self.odometry.pose.pose.orientation.w]], device=self.device)
        self.base_lin_vel = self.quat_rotate_inverse(self.base_quat, base_lin_vel_list)
        self.base_ang_vel = self.quat_rotate_inverse(self.base_quat, base_ang_vel_list)
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

        self.obs_history = torch.cat((self.obs_history[:, 70:], self.mit_observations), dim=-1)
        self.latent = self.adaptation_module.forward(self.obs_history.cpu())
        print("latent: ",self.latent)

        self.mit_actions = self.body_module.forward(torch.cat((self.obs_history.cpu(), self.latent), dim=-1))
        

        self.obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        
        self.prev_actions = self.mit_actions.to(self.device).clone().detach()

        self.mit_actions = torch.clip(self.mit_actions, -self.clip_actions, self.clip_actions).to(self.device)


        # ------------------- Publish Joint Torques ------------------- #

        torques = self.compute_torques(self.mit_actions * self.action_scale)

        self.publish_trajectory_torques(torques.tolist()[0])
        
        end_time = time.time()
        duration = end_time - start_time
        # print(duration)
        

        return self.obs_buf

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
        # print(torques, "\n -------------- \n")
        # self.joint_torque_publisher.publish(arr)


    def compute_torques(self, actions):

        torques = self.p_gains*(actions + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    


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

