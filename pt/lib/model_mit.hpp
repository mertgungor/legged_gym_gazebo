#ifndef MODEL_MIT_HPP
#define MODEL_MIT_HPP

#include <torch/script.h>
#include <iostream>
#include <string>

struct ModelMITParams {

    float num_observations;
    float damping;
    float stiffness;
    float action_scale;
    float num_of_dofs;
    float lin_vel_scale;
    float ang_vel_scale;
    float dof_pos_scale;
    float dof_vel_scale;

    float clip_obs;
    float clip_actions;
    torch::Tensor torque_limits;
    torch::Tensor d_gains;
    torch::Tensor p_gains;
    torch::Tensor commands_scale;
    torch::Tensor default_dof_pos;

    ModelMITParams(
        float num_observations,
        float damping,
        float stiffness,
        float action_scale,
        float num_of_dofs,
        float lin_vel_scale,
        float ang_vel_scale,
        float dof_pos_scale,
        float dof_vel_scale,
        float clip_obs,
        float clip_actions,
        torch::Tensor torque_limits,
        torch::Tensor d_gains,
        torch::Tensor p_gains,
        torch::Tensor commands_scale,
        torch::Tensor default_dof_pos);
    
    ModelMITParams();
    
};

class ModelMIT {
    public:

        torch::Tensor forward();
        torch::Tensor compute_torques(torch::Tensor actions);
        torch::Tensor quat_rotate_inverse(torch::Tensor q, torch::Tensor v);
        torch::Tensor compute_observation();
        void update_observations(torch::Tensor lin_vel, torch::Tensor ang_vel, torch::Tensor commands, torch::Tensor base_quat, torch::Tensor dof_pos, torch::Tensor dof_vel);
        void init_observations();
        void init_params();
        ModelMIT(std::string model_path, ModelMITParams params);
        ModelMITParams params;

    private:


        // observation buffer
        torch::jit::script::Module module;
        torch::Tensor lin_vel;           
        torch::Tensor ang_vel;           
        torch::Tensor commands;          
        torch::Tensor dof_pos;           
        torch::Tensor dof_vel;           
        torch::Tensor actions;

        torch::Tensor default_dof_pos; 
        torch::Tensor base_quat; 
        torch::Tensor gravity_vec; 

        float num_observations;
        float clip_obs;
        float clip_actions;
        torch::Tensor torque_limits;
        float damping;
        float stiffness;
        torch::Tensor d_gains;
        torch::Tensor p_gains;
        float action_scale;
        float num_of_dofs;
        torch::Tensor commands_scale;
        torch::Tensor obs_history;

        // scales
        float lin_vel_scale;
        float ang_vel_scale;
        float dof_pos_scale;
        float dof_vel_scale;
        float height_measurements_scale;


        std::string joint_names[12];
        
};

#endif // MODEL_MIT_HPP