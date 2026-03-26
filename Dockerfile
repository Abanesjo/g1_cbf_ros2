FROM osrf/ros:humble-desktop-full

RUN apt update && apt upgrade -y

# ROS2 and build deps
RUN apt install -y \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-rosbag2-cpp \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-pinocchio \
    libyaml-cpp-dev \
    libeigen3-dev \
    python3-pip \
    tmux

RUN pip3 install osqp "jax[cpu]" "numpy<2"

# Workspace setup
WORKDIR /workspace

SHELL ["/bin/bash", "-c"]

COPY entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
