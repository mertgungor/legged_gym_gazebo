#include "../lib/model.hpp"
#include "../lib/unitree_custom.hpp"


int main(void)
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


    Model model("/home/mert/pt-cpp/models/policy_1.pt", a1_params);

    std::cout << "Communication level is set to LOW-level." << std::endl
              << "WARNING: Make sure the robot is hung up." << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    UnitreeCustom Unitreecustom(LOWLEVEL, model);
    // InitEnvironment();
    LoopFunc loop_control("control_loop", Unitreecustom.dt,    boost::bind(&UnitreeCustom::RobotControl, &Unitreecustom));
    LoopFunc loop_udpSend("udp_send",     Unitreecustom.dt, 3, boost::bind(&UnitreeCustom::UDPSend,      &Unitreecustom));
    LoopFunc loop_udpRecv("udp_recv",     Unitreecustom.dt, 3, boost::bind(&UnitreeCustom::UDPRecv,      &Unitreecustom));

    loop_udpSend.start();
    loop_udpRecv.start();
    loop_control.start();

    while(1){
        sleep(10);
    };

    return 0; 
}