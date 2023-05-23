#include "../lib/model.hpp"
#include "../lib/unitree_custom.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

class A1Ros : public rclcpp::Node
{

public:

    A1Ros(Model model) : model(model), Node("a1_ros"){

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

    }

private:

    Model model;

    geometry_msgs::msg::Twist::SharedPtr cmd_vel;
    nav_msgs::msg::Odometry::SharedPtr odom;
    sensor_msgs::msg::JointState::SharedPtr joint_state;
    std_msgs::msg::Float64MultiArray::SharedPtr joint_command;

    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_command_pub;

    void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg){
        cmd_vel = msg;
    }

    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg){
        odom = msg;
    }

    void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg){
        joint_state = msg;
    }

};

int main(int argc, char** argv)

{
    std::string share_directory = ament_index_cpp::get_package_share_directory("pt");

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



    Model model(share_directory + "/models/policy_1.pt", a1_params);

    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<A1Ros>(model));
    rclcpp::shutdown();


    return 0; 
}