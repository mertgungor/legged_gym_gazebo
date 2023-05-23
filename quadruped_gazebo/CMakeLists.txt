cmake_minimum_required(VERSION 3.5)
project(quadruped_gazebo)


find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch worlds meshes urdf rviz config
  DESTINATION share/${PROJECT_NAME} 
)

ament_package()
