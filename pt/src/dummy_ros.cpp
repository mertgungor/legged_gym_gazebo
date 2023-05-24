#include "rclcpp/rclcpp.hpp"

class MyNode : public rclcpp::Node {
public:
  MyNode() : Node("my_node") {
    timer_ = create_wall_timer(std::chrono::seconds(1), std::bind(&MyNode::timerCallback, this));
  }

private:
  void timerCallback() {
    RCLCPP_INFO(get_logger(), "Timer callback");
  }

  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MyNode>();

  rclcpp::spin(node);
  rclcpp::shutdown();

  return 0;
}
