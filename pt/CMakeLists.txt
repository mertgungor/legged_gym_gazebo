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

add_library(model_mit lib/model_mit.cpp lib/model_mit.hpp)
target_link_libraries(model_mit "${TORCH_LIBRARIES}")
set_property(TARGET model_mit PROPERTY CXX_STANDARD 14)

add_executable(a1 src/a1.cpp)
target_link_libraries(a1 model unitree_custom ${EXTRA_LIBS} "${TORCH_LIBRARIES}")
set_property(TARGET a1 PROPERTY CXX_STANDARD 14)
# ament_target_dependencies(a1 rclcpp Torch)


add_executable(dummy_ros src/dummy_ros.cpp)
target_link_libraries(dummy_ros )
set_property(TARGET dummy_ros PROPERTY CXX_STANDARD 14)
ament_target_dependencies(dummy_ros rclcpp )

# a1 ros ---------------------
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

# aselsan ros ---------------------

add_executable(aselsan_ros src/aselsan_ros.cpp)

target_link_libraries(
  aselsan_ros model 
  unitree_custom 
  ament_index_cpp::ament_index_cpp 
  ${rclcpp_LIBRARIES} 
  ${EXTRA_LIBS} 
  "${TORCH_LIBRARIES}")

set_property(TARGET a1 PROPERTY CXX_STANDARD 14)
ament_target_dependencies(
  aselsan_ros 
  geometry_msgs 
  nav_msgs 
  rclcpp 
  Torch
  sensor_msgs
  std_msgs )

# a1 MIT ---------------------

add_executable(a1_mit src/a1_mit.cpp)

target_link_libraries(
  a1_mit model 
  unitree_custom 
  ament_index_cpp::ament_index_cpp 
  ${rclcpp_LIBRARIES} 
  ${EXTRA_LIBS} 
  "${TORCH_LIBRARIES}")

set_property(TARGET a1_mit PROPERTY CXX_STANDARD 14)
ament_target_dependencies(
  a1_mit 
  geometry_msgs 
  nav_msgs 
  rclcpp 
  Torch
  sensor_msgs
  std_msgs )
#  ---------------------
install(TARGETS
  dummy_ros
  a1
  a1_ros
  a1_mit
  aselsan_ros
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
