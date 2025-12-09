cmake_minimum_required(VERSION 3.8)
project(opponent_tracker)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

ament_python_install_package(opponent_tracker)

install(PROGRAMS
  opponent_tracker/opponent_odom_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
