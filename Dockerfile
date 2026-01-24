# ------------------------------------------------------------
# GLOBAL ARGS
# ------------------------------------------------------------
ARG ROS2_DISTRO=humble


# ------------------------------------------------------------
# STAGE 1: BASE (System Setup)
# ------------------------------------------------------------
# Use official ROS base image (multi-arch: amd64, arm64v8)
FROM ros:${ROS2_DISTRO} AS ros2_base

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        nano \
        build-essential \
        tmux \
        tmuxinator \
        ranger \
        curl \
        openssh-client \
        iputils-ping \
        libnss-mdns \
        libboost-python-dev \
        python3-pip \
        python3-tk \
        libnss-mdns \
        avahi-utils \
        dbus \
    && sed -i 's/^hosts:.*/hosts:          files dns mdns4_minimal/' /etc/nsswitch.conf \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip


# ------------------------------------------------------------
# STAGE 2: CORE (The main package and dependencies)
# ------------------------------------------------------------
FROM ros2_base AS core

# Re-declare ARG variables after FROM for use in subsequent layers
ARG HOME=/root
ARG ROS2_WS=${HOME}/ros2_ws
ARG OTHER_WS=${HOME}/other_ws
ARG ROS_DOMAIN_ID=60

# Install Python dependencies
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-geometric==2.7.0
RUN pip install \
    matplotlib==3.10.7 \
    scipy==1.15.3 \
    networkx==3.3 \
    pandas==2.3.3 \
    plotly==6.4.0 \
    prettytable==3.16.0 \
    shortuuid==1.0.13 \
    digi-xbee \
    colorlog

# Install ROS packages
RUN apt install -y \
    ros-$ROS_DISTRO-foxglove-bridge \
    ros-$ROS_DISTRO-rosbag2-storage-mcap

# Copy configuration files
COPY docker/to_copy/tmux.conf ${HOME}/.tmux.conf
COPY docker/to_copy/aliases ${HOME}/.bash_aliases
COPY docker/to_copy/nanorc ${HOME}/.nanorc
COPY docker/to_copy/ranger ${HOME}/.config/ranger/rc.conf

# Set up workspaces
RUN mkdir -p ${OTHER_WS}
RUN mkdir -p ${ROS2_WS}/src
RUN mkdir -p ${ROS2_WS}/bags

# Clone/copy general repository dependencies
WORKDIR ${OTHER_WS}
RUN git clone https://github.com/mkrizmancic/my_graphs_dataset.git

# Build and install general packages
RUN cd ${OTHER_WS}/my_graphs_dataset && pip install -e .

# Clone the example ROS2 package
WORKDIR ${ROS2_WS}/src
RUN git clone https://github.com/mkrizmancic/ros2_dec_gnn.git
RUN cd ros2_dec_gnn && git checkout app_mids

# Build ROS2 workspace
WORKDIR ${ROS2_WS}
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build --symlink-install"

# Add ROS2 workspace setup to .bashrc
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> $HOME/.bashrc
RUN echo "source $HOME/ros2_ws/install/setup.bash" >> $HOME/.bashrc
RUN echo "export ROS_DOMAIN_ID=${ROS_DOMAIN_ID}" >> $HOME/.bashrc

# Copy the current project files into the container
COPY . ${OTHER_WS}/xbee_dec_gnn

# ------------------------------------------------------------
# STAGE 3: FINAL (Mix in optional additions)
# ------------------------------------------------------------
FROM core AS final

ARG ENABLE_DEV=false
RUN if [ "$ENABLE_DEV" = "true" ]; then \
pip install ipython ipykernel nbformat && \
pip install tqdm && \
pip install tabulate; \
fi

ARG ENABLE_LED=false
RUN if [ "$ENABLE_LED" = "true" ]; then \
apt-get install -y scons && \
pip install rpi_ws281x && \
cd ${OTHER_WS} && \
git clone https://github.com/jgarff/rpi_ws281x.git && \
cd ${OTHER_WS}/rpi_ws281x && \
sed -i 's/^#define WIDTH.*/#define WIDTH                   8/' main.c && \
sed -i 's/^#define HEIGHT.*/#define HEIGHT                  4/' main.c && \
scons; \
fi

# Clean up apt cache for smaller image
RUN rm -rf /var/lib/apt/lists/*

# Final settings
WORKDIR ${HOME}
ENTRYPOINT []
CMD ["/bin/bash"]

