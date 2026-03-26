#!/bin/bash
source /opt/ros/humble/setup.bash
cd /workspace
pip3 install /workspace/src/unitree_ros2/g1_cbf_ros2/dependencies/dpax/
colcon build --symlink-install --parallel-workers $(( $(nproc) / 2 ))
source /workspace/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///workspace/src/unitree_ros2/cyclonedds.xml
echo "source /workspace/install/setup.bash" >> ~/.bashrc
exec bash
