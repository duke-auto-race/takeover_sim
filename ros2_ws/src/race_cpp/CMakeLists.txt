cmake_minimum_required(VERSION 3.8)
project(race_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclpy REQUIRED)
find_package(visualization_msgs REQUIRED)
# find_package(scout_msgs REQUIRED)
find_package(osqp REQUIRED)
find_package(OsqpEigen REQUIRED)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

include_directories(include)

# EXECUTABLES HERE
###################################################################

# race_sim_server
add_executable(race_sim_server_node
	src/race_sim_server_node.cpp
    src/race_sim_server.cpp
)
ament_target_dependencies(race_sim_server_node
	rclcpp 
	geometry_msgs
	sensor_msgs
	mavros_msgs
	visualization_msgs
)
install(TARGETS race_sim_server_node DESTINATION lib/${PROJECT_NAME})

###################################################################
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()