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
        
        
        if self.visualize_height:
            
            self.fig = plt.figure(figsize=(10, 8))
            self.ax  = self.fig.add_subplot(111, projection='3d')
            
            self.ax.set_xlim(-5, 5)
            self.ax.set_ylim(-5, 5)
            self.ax.set_zlim(0, 1.5)
            self.ax.grid(linewidth=2)
            
            self.point, = self.ax.plot([], [], [], 'ro')

            

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
            
        if self.params["num_observations"] == 235:
            
            self.measured_points_x = torch.arange(-0.8, 0.9, 0.1) # 1mx1.6m rectangle (without center line)
            self.measured_points_y = torch.arange(-0.5, 0.6, 0.1)

            self.grid_x, self.grid_y = torch.meshgrid(
                torch.tensor(self.measured_points_y, device=self.device, requires_grad=False).clone().detach(), 
                torch.tensor(self.measured_points_x, device=self.device, requires_grad=False).clone().detach()
                )
            
            self.num_envs = 1

            self.num_height_points = self.grid_x.numel()
            self.points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
            self.points[:, :, 0] = self.grid_x.flatten()
            self.points[:, :, 1] = self.grid_y.flatten()
            # print(self.points[:, :, 0][0].numpy())

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

        self.actions           = torch.tensor([[0.0] * self.num_of_dofs], device=self.device)
        self.commands          = torch.tensor([self.current_command], device=self.device)
        self.base_quat         = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self.gravity_vec       = torch.tensor([[0.0, 0.0, -1.0]], device=self.device)
        self.projected_gravity = torch.tensor([[0.0, 0.0, -9.81]], device=self.device)   

        self.obs_buf = torch.tensor([], device=self.device)

        self.obs_scales = {
            "lin_vel": self.params["lin_vel"],
            "ang_vel": self.params["ang_vel"],
            "dof_pos": self.params["dof_pos"],
            "dof_vel": self.params["dof_vel"],
            "height_measurements" : 5.0
        } 


        self.joint_names = self.params["joint_names"]

        self.commands_scale =  torch.tensor([self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]], device=self.device, requires_grad=False,)
        

        models_path = os.path.join(install_dir, "models")
        model_path = os.path.join(models_path, self.params["model_path"]) 
            
        
        self.policy = initialize_runner(model_path, self.params)

        print("Waiting for joint states and odometry...")
        while self.dof_pos.size() == torch.Size([0]) or self.dof_vel.size() == torch.Size([0]):
            rclpy.spin_once(self, timeout_sec=0.1)
        print("Joint states and odometry received.")

        self.create_timer(0.002, self.compute_observation)  
                                 
    def cmd_vel_callback(self, msg):
        self.current_command = [msg.linear.x, msg.linear.y, msg.angular.z]      
        self.commands        = torch.tensor([self.current_command], device=self.device)          

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

        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales["lin_vel"],
                                    self.base_ang_vel  * self.obs_scales["ang_vel"],
                                    self.projected_gravity,
                                    self.commands[:3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
                                    self.dof_vel * self.obs_scales["dof_vel"],
                                    # torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=self.device)
                                    self.actions
                                    ),dim=-1)
        
        # print(self.obs_buf)

        if self.params["num_observations"] == 235:
            
            # displacement = torch.tensor([[
            #     self.odometry.pose.pose.position.x,
            #     self.odometry.pose.pose.position.y,
            #     self.odometry.pose.pose.position.z
            # ]], device=self.device)
            
            # points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.points) + (displacement[:, :3]).unsqueeze(1)
            
            # if self.visualize_height:
            #     self.visualize(points)
            
            measured_heights = torch.tensor([[0.0] * 187], device=self.device)

            heights = torch.clip(self.odometry.pose.pose.position.z - 0.5 - measured_heights, -1, 1.) * self.obs_scales["height_measurements"]
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        
        self.obs_buf = torch.clip(self.obs_buf, -self.clip_obs, self.clip_obs)
        
        self.actions = self.policy(self.obs_buf)

        self.actions = torch.clip(self.actions, -self.clip_actions, self.clip_actions).to(self.device)

        # ------------------- Publish Joint Torques ------------------- #

        torques = self.compute_torques(self.actions * self.action_scale)

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
        self.joint_torque_publisher.publish(arr)


    def compute_torques(self, actions):

        torques = self.p_gains*(actions + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def visualize(self, points):
        self.point.set_data(points[:, :, 0][0].cpu().numpy(), points[:, :, 1][0].cpu().numpy())
        self.point.set_3d_properties([0])
        self.fig.canvas.draw()
        plt.pause(0.001)

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

