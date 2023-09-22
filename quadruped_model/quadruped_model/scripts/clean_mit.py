#!/usr/bin/env python3

import os
import sys
import time
import rclpy
import yaml
import torch

from isaacgym.torch_utils         import quat_apply, normalize
from rclpy.node                   import Node
from sensor_msgs.msg              import JointState
from nav_msgs.msg                 import Odometry
from quadruped_model.scripts.play import initialize_runner
from sensor_msgs.msg              import JointState
from std_msgs.msg                 import Float64MultiArray
from geometry_msgs.msg            import Twist
from ament_index_python.packages  import get_package_share_directory
from yaml.loader                  import SafeLoader

import numpy             as np
import matplotlib.pyplot as plt

# @ torch.jit.script

def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)

class ObservationNode(Node):
    def __init__(self):

        super().__init__('observation_node')
        
        self.device = 'cpu'
        self.joint_pos_err_last_last = torch.zeros((1, 12), device=self.device)
        self.joint_pos_err_last      = torch.zeros((1, 12), device=self.device)

        self.joint_vel_last_last = torch.zeros((1, 12), device=self.device)
        self.joint_vel_last      = torch.zeros((1, 12), device=self.device)

        self.joint_state = JointState()
        self.odometry = Odometry()

        self.controller = "joint_group_effort_controller"
        self.joint_trajectory_topic = "/{}/joint_trajectory".format(self.controller)
        self.joint_torque_topic = "/{}/commands".format(self.controller)

        self.join_state_pose_tensor = torch.zeros((12), device=self.device)
        self.join_state_vel_tensor  = torch.zeros((12), device=self.device)

        self.clip_obs = 100
        self.clip_actions = 100           # hip, thigh, calf
        self.torque_limits = torch.tensor([23.7, 23.7,  35.55,  # front left
                                           23.7, 23.7,  35.55,  # front right
                                           23.7, 23.7,  35.55,  # rear  left
                                           23.7, 23.7,  35.55], # rear  right
                                           device=self.device)

        self.damping = 0.8
        self.stiffness = 50.0
        self.d_gains = torch.tensor([self.damping] * 12, device=self.device)
        self.p_gains = torch.tensor([self.stiffness] * 12, device=self.device)
        self.action_scale = 0.25

        self.dof_pos      = torch.tensor([[] * self.num_of_dofs], device=self.device)
        self.dof_vel      = torch.tensor([[] * self.num_of_dofs], device=self.device)

        self.joint_torque_publisher = self.create_publisher(   Float64MultiArray, self.joint_torque_topic, 10)
        self.odometry_subscriber    = self.create_subscription(Odometry,          'odom/ground_truth',     self.odometry_callback,    10)
        self.joint_state_subscriber = self.create_subscription(JointState,        'joint_states',          self.joint_state_callback, 10)
        self.cmd_vel_subscriber     = self.create_subscription(Twist,             'cmd_vel',               self.cmd_vel_callback,     10)
 
        self.num_of_dofs = 12
                                                # hip,    thigh,  calf
        self.default_dof_pos =  torch.tensor([ [ 0.1000,  0.8000, -1.5000, # front left
                                                -0.1000,  0.8000, -1.5000, # front right
                                                 0.1000,  1.0000, -1.5000, # rear  left
                                                -0.1000,  1.0000, -1.5000  # rear  right
                                                ]], device=self.device)                   
        
        self.dof_vel      = torch.tensor([[0.0]*12], device=self.device)
        self.dof_pos      = self.default_dof_pos

        self.current_command = [0.0, 0.0, 0.0]

        self.mit_actions       = torch.tensor([[0.0] * self.num_of_dofs], device=self.device)
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

        self.obs_history = torch.zeros(1, 30*70, dtype=torch.float, device=self.device, requires_grad=False)

        actuator_path = "/home/mert/ros-pt/src/quadruped_model/models/a1_flat/unitree_go1.pt"
        self.actuator_network = torch.jit.load(actuator_path).to(self.device)

        self.joint_names = [
            "FL_hip_joint"  ,
            "FL_thigh_joint",
            "FL_calf_joint" ,
            "FR_hip_joint"  ,
            "FR_thigh_joint",
            "FR_calf_joint" ,
            "RL_hip_joint"  ,
            "RL_thigh_joint",
            "RL_calf_joint" ,
            "RR_hip_joint"  ,
            "RR_thigh_joint",
            "RR_calf_joint" ,
            ]
        
        self.mit_commands_scale = torch.tensor([self.params["lin_vel"],              self.params["lin_vel"], self.params["ang_vel"],#"lin_vel"             
                                                self.params["body_height_cmd"],      self.params["gait_freq_cmd"],#"body_height_cmd"     
                                                self.params["gait_phase_cmd"],       self.params["gait_phase_cmd"],#"gait_phase_cmd"      
                                                self.params["gait_phase_cmd"],       self.params["gait_phase_cmd"],#"gait_phase_cmd"      
                                                self.params["footswing_height_cmd"], self.params["body_pitch_cmd"],#"footswing_height_cmd"
                                                self.params["body_roll_cmd"],        self.params["stance_width_cmd"],#"body_roll_cmd"       
                                                self.params["stance_length_cmd"],    self.params["aux_reward_cmd"]],                  #"stance_length_cmd"   
                                                device=self.device, requires_grad=False, )


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