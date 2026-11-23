cmake_minimum_required(VERSION 3.0.2)
project(controller_node)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs duckietown_msgs)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/duckiebot_controller_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
