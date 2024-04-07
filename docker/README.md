# Docker Usage

## Install Nvidia Container Tookit

/etc/nvidia-container-runtime/config.toml

```
ls -al /dev/nvidia*
```

```
user = "root:vglusers"
```

## Build Docker Image

Build gpu accleated ROS2 foxy in Docker

```
docker build -t ghcr.io/donghee/wearable_robot_eval:foxy_nvidia_novnc . -f Dockerfile.nvidia.novnc
```

## Run Container

```
docker run -it \
    --gpus all \
    --privileged \
    --volume="/tmp/.X11-unix/X0:/tmp/.X11-unix/X0:rw" \
    --rm \
    -p 80:6080 \
    --workdir="/root" \
    ghcr.io/donghee/wearable_robot_eval:foxy_nvidia_novnc \
    bash -c "/bootstrap.sh && /opt/TurboVNC/bin/vncserver -wm LXDE -SecurityTypes None :1 && /opt/noVNC/utils/websockify/run --verbose --web=/opt/noVNC/ 6080 127.0.0.1:5901"
```
