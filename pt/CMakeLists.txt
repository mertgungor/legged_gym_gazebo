cmake_minimum_required(VERSION 3.5)
project(pt)

set(CMAKE_INSTALL_RPATH "$ENV{HOME}/Downloads/libtorch/lib")
set(EXTRA_LIBS -pthread libunitree_legged_sdk_amd64.so lcm)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")


find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(Torch REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)




include_directories(
  ~/Downloads/unitree_legged_sdk/include
)

link_directories(
  lib
  ~/Downloads/unitree_legged_sdk/lib
)


add_library(unitree_custom lib/unitree_custom.cpp lib/unitree_custom.hpp)
target_link_libraries(unitree_custom ${EXTRA_LIBS} "${TORCH_LIBRARIES}")
set_property(TARGET unitree_custom PROPERTY CXX_STANDARD 14)

add_library(model lib/model.cpp lib/model.hpp)
target_link_libraries(model "${TORCH_LIBRARIES}")
set_property(TARGET model PROPERTY CXX_STANDARD 14)

add_executable(a1 src/a1.cpp)
target_link_libraries(a1 model unitree_custom ${EXTRA_LIBS} "${TORCH_LIBRARIES}")
set_property(TARGET a1 PROPERTY CXX_STANDARD 14)
# ament_target_dependencies(a1 rclcpp Torch)



add_executable(a1_ros src/a1_ros.cpp)

target_link_libraries(
  a1_ros model 
  unitree_custom 
  ament_index_cpp::ament_index_cpp 
  ${rclcpp_LIBRARIES} 
  ${EXTRA_LIBS} 
  "${TORCH_LIBRARIES}")

set_property(TARGET a1 PROPERTY CXX_STANDARD 14)
ament_target_dependencies(
  a1_ros 
  geometry_msgs 
  nav_msgs 
  rclcpp 
  Torch
  sensor_msgs
  std_msgs )

install(TARGETS

  a1
  a1_ros
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
