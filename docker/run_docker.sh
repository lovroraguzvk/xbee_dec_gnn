#!/bin/bash

IMAGE_NAME=${1:-xbee_gnn_img}
CONTAINER_NAME=${2:-xbee_gnn_cont}

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Auto-detect if running on Raspberry Pi
IS_RPI=false
if [ -f /proc/cpuinfo ] && grep -q "Raspberry Pi" /proc/cpuinfo; then
  IS_RPI=true
elif [ -f /sys/firmware/devicetree/base/model ] && grep -q "Raspberry Pi" /sys/firmware/devicetree/base/model; then
  IS_RPI=true
fi

if [ "$IS_RPI" = true ]; then
  echo "Running in RPI mode..."
  docker run \
    -it \
    --network host \
    --ipc host \
    --privileged \
    --volume /var/run/dbus:/var/run/dbus \
    --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
    --volume $SCRIPT_DIR/volumes/datasets:/root/resources/data \
    --volume $SCRIPT_DIR/volumes/models:/root/resources/models \
    --env TERM=xterm-256color \
    --name $CONTAINER_NAME \
    $IMAGE_NAME
else
  echo "Running in default mode..."
  docker run \
    -it \
    --network host \
    --device /dev/serial/by-id/usb-FTDI_FT231X_USB_UART_D30AY52N-if00-port0:/dev/ttyUSB0 \
    --ipc host \
    --volume ~/.ssh/ssh_auth_sock:/ssh-agent \
    --volume ~/.ssh:/root/.ssh \
    --volume /var/run/dbus:/var/run/dbus \
    --volume /var/run/avahi-daemon/socket:/var/run/avahi-daemon/socket \
    --volume $SCRIPT_DIR/volumes/datasets:/root/resources/data \
    --volume $SCRIPT_DIR/volumes/models:/root/resources/models \
    --volume /tmp/.x11-unix:/tmp/.x11-unix \
    --env SSH_AUTH_SOCK=/ssh-agent \
    --env DISPLAY=$DISPLAY \
    --env TERM=xterm-256color \
    --name $CONTAINER_NAME \
    $IMAGE_NAME
fi
