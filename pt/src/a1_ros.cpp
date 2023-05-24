#include "../lib/model.hpp"
#include "../lib/unitree_custom.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <iostream>
#include <vector>
#include <chrono>

class A1Ros : public rclcpp::Node
{

public:

    A1Ros(Model model) : model(model), Node("a1_ros"){

        cmd_vel             = std::make_shared<geometry_msgs::msg::Twist>();
        joint_command       = std::make_shared<std_msgs::msg::Float64MultiArray>();
        joint_command->data = std::vector<double>(12, 0.0);

        // Create subscriber for cmd_vel, odom, and joint_state

        cmd_vel_sub = this->create_subscription<geometry_msgs::msg::Twist>(
            "cmd_vel", 10, std::bind(&A1Ros::cmd_vel_callback, this, std::placeholders::_1));

        odom_sub = this->create_subscription<nav_msgs::msg::Odometry>(
            "odom/ground_truth", 10, std::bind(&A1Ros::odom_callback, this, std::placeholders::_1));

        joint_state_sub = this->create_subscription<sensor_msgs::msg::JointState>(
            "joint_states", 10, std::bind(&A1Ros::joint_state_callback, this, std::placeholders::_1));

        // Create publisher for joint_command

        joint_command_pub = this->create_publisher<std_msgs::msg::Float64MultiArray>(
            "joint_group_effort_controller/commands", 10);

        // Wait for topics to be ready

        rclcpp::sleep_for(std::chrono::seconds(2));

        printf("Initializing timer...\n");


        timer_ = create_wall_timer(std::chrono::milliseconds(2), std::bind(&A1Ros::run_model, this));
        

    }

private:

    Model model;

    rclcpp::TimerBase::SharedPtr timer_;

    geometry_msgs::msg::Twist::SharedPtr        cmd_vel;
    nav_msgs::msg::Odometry::SharedPtr          odom;
    sensor_msgs::msg::JointState::SharedPtr     joint_state;
    std_msgs::msg::Float64MultiArray::SharedPtr joint_command;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr    cmd_vel_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr      odom_sub;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_pub;

    std::vector<double> joint_state_poses;
    std::vector<double> joint_state_velocities;

    std::map<std::string, double> joint_pose_map;
    std::map<std::string, double> joint_vels_map;

    torch::Tensor dof_pos = torch::zeros({12});
    torch::Tensor dof_vel = torch::zeros({12});
    torch::Tensor torques = model.compute_torques(model.forward());

    std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();



    std::string joint_names[12] = {
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
        "RR_calf_joint" };

    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg){
        cmd_vel = msg;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg){
        odom = msg;
    }

    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg){
        joint_state = msg;

        std::vector<std::string> joint_state_names = msg->name;
        joint_state_poses      = msg->position;
        joint_state_velocities = msg->velocity;

        joint_state->velocity = std::vector<double>(12, 0.0);

        for (int i = 0; i < joint_state_names.size(); i++){
            joint_pose_map[joint_state_names[i]] = joint_state_poses[i];
            joint_vels_map[joint_state_names[i]] = joint_state_velocities[i];
        }

        for (int i = 0; i < 12; i++){
            joint_state->position[i] = joint_pose_map[joint_names[i]];
            joint_state->velocity[i] = joint_vels_map[joint_names[i]];
        }

        for (int i = 0; i < 12; i++){
            dof_pos[i] = joint_state->position[i];
            dof_vel[i] = joint_state->velocity[i];
        }
        
    }

    void run_model(){
        printf("Running model...\n");

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "Execution time: " << duration << " microseconds" << std::endl;
        start = std::chrono::high_resolution_clock::now();

        torch::Tensor lin_vel     = torch::tensor({{odom->twist.twist.linear.x, odom->twist.twist.linear.y, odom->twist.twist.linear.z}});
        torch::Tensor ang_vel     = torch::tensor({{odom->twist.twist.angular.x, odom->twist.twist.angular.y, odom->twist.twist.angular.z}});
        torch::Tensor command     = torch::tensor({{cmd_vel->linear.x, cmd_vel->linear.y, cmd_vel->angular.z}});
        torch::Tensor orientation = torch::tensor({{odom->pose.pose.orientation.x, odom->pose.pose.orientation.y, odom->pose.pose.orientation.z, odom->pose.pose.orientation.w}});
        
        printf("Tensors initialized...\n");

        // printf("lin_vel size     : %d\n", lin_vel.sizes()[0]);
        // printf("ang_vel size     : %d\n", ang_vel.sizes()[0]);
        // printf("command size     : %d\n", command.sizes()[0]);
        // printf("orientation size : %d\n", orientation.sizes()[0]);
        // printf("dof_pos size     : %d\n", dof_pos.sizes()[0]);
        // printf("dof_vel size     : %d\n", dof_vel.sizes()[0]);

        for(int i = 0; i < 12; i++){
            std::cout << "Torques " << joint_names[i] << " : " << torques[0][i].item<float>() << std::endl;
        }

        // // calf ofset
        // dof_pos[2]  -= 0.9;
        // dof_pos[5]  -= 0.9;
        // dof_pos[8]  -= 0.9;
        // dof_pos[11] -= 0.9;

        // // tigh ofset
        // dof_pos[1]  += 1.5;
        // dof_pos[4]  += 1.5;
        // dof_pos[7]  += 1.5;
        // dof_pos[10] += 1.5;

        model.update_observations(
            lin_vel              ,
            ang_vel              ,
            command              ,
            orientation          ,
            dof_pos              ,
            dof_vel.view({1, 12})
            );
        
        
        torques = model.compute_torques(model.forward());

        for(int i = 0; i < joint_command->data.size(); i++){
            joint_command->data[i] = torques[0][i].item<float>();
        }

        joint_command_pub->publish(*joint_command); 

    }


};

int main(int argc, char** argv)

{

    ModelParams a1_params;
    a1_params.num_observations = 48;
    a1_params.clip_obs         = 100.0;
    a1_params.clip_actions     = 100.0;                                             
    a1_params.damping          = 1;
    a1_params.stiffness        = 40;
    a1_params.d_gains          = torch::ones(12)*a1_params.damping;
    a1_params.p_gains          = torch::ones(12)*a1_params.stiffness;
    a1_params.action_scale     = 0.25;
    a1_params.num_of_dofs      = 12;
    a1_params.lin_vel_scale    = 2.0;
    a1_params.ang_vel_scale    = 0.25;
    a1_params.dof_pos_scale    = 1.0;
    a1_params.dof_vel_scale    = 0.05;
    a1_params.commands_scale   = torch::tensor({a1_params.lin_vel_scale, a1_params.lin_vel_scale, a1_params.ang_vel_scale});

                                               //hip, thigh, calf
    a1_params.torque_limits    = torch::tensor({{20.0, 55.0, 55.0,   // front left
                                                 20.0, 55.0, 55.0,   // front right
                                                 20.0, 55.0, 55.0,   // rear  left
                                                 20.0, 55.0, 55.0 }}); // rear  right

                                                 
    a1_params.default_dof_pos  = torch::tensor({{ 0.1000,  0.8000, -1.5000,    
                                                 -0.1000,  0.8000, -1.5000,    
                                                  0.1000,  1.0000, -1.5000,    
                                                 -0.1000,  1.0000, -1.5000 }});   

    std::string share_directory = ament_index_cpp::get_package_share_directory("pt");

    Model model(share_directory + "/models/policy_1.pt", a1_params);

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<A1Ros>(model));
    rclcpp::shutdown();


    return 0; 
}