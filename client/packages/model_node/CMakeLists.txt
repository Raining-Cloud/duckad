cmake_minimum_required(VERSION 3.0.2)
project(model_node)

find_package(catkin REQUIRED COMPONENTS rospy sensor_msgs std_msgs)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/model_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES scripts/onnx_model.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY assets/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/assets
)
