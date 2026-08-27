cmake_minimum_required(VERSION 2.8.3)
project(multi_robot_scenario)

find_package(catkin REQUIRED COMPONENTS xacro)

catkin_package(
  CATKIN_DEPENDS xacro
)

install(
  DIRECTORY launch meshes xacro
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
