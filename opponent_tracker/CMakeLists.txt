cmake_minimum_required(VERSION 3.8)
project(opponent_tracker)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  opponent_tracker/opponent_odom_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  evaluate_opponent.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME evaluate_opponent
)

install(DIRECTORY
  launch
  config
  rviz
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
