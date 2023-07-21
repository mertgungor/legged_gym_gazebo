cmake_minimum_required(VERSION 3.8)
project(quadruped_model)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(PCL_DIR "/usr/lib/x86_64-linux-gnu/cmake/pcl/PCLConfig.cmake")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(tf2_ros REQUIRED)



ament_python_install_package(${PROJECT_NAME})

add_executable(tf2_listener src/tf2_listener.cpp)
ament_target_dependencies(tf2_listener rclcpp tf2_ros sensor_msgs pcl_ros tf2)

add_executable(dummy src/dummy.cpp)
ament_target_dependencies(dummy rclcpp tf2_ros sensor_msgs pcl_ros tf2)

# add_executable(pointcloud_subscriber src/pointcloud_subscriber.cpp)
# ament_target_dependencies(pointcloud_subscriber rclcpp sensor_msgs pcl_ros)

install(PROGRAMS
  ${PROJECT_NAME}/scripts/model_torque.py


  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY models config launch maps rviz
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS
  # pointcloud_subscriber
  tf2_listener
  dummy
  DESTINATION lib/${PROJECT_NAME})

ament_package()
