#!/bin/sh

xhost +local:root

# no gpu
docker run -it \
    --privileged \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="${PWD}/..:/One-Shot_Free-View_Neural_Talking_Head_Synthesis" \
    --volume="${PWD}/data:/data" \
    --volume="${PWD}/checkpoint:/checkpoint" \
    --volume="${PWD}/Train_data:/Train_data" \
    --volume="${PWD}/Test_data:/Test_data" \
    --name one-shot-free-view-neural-talking-head-synthesis \
    --restart unless-stopped \
    --gpus all \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -p 8889:8889 \
    ghcr.io/donghee/one-shot-free-view-neural-talking-head-synthesis:cuda11.0.3-devel-ubuntu20.04 \
    bash
    #/One-Shot_Free-View_Neural_Talking_Head_Synthesis/run-web.sh
    #python3 /One-Shot_Free-View_Neural_Talking_Head_Synthesis/demo.py --result_video /data/result.mp4 --config /checkpoint/vox-256-spade.yaml --checkpoint /checkpoint/00000189-checkpoint.pth.zip --source_image /data/man.png --driving_video /data/man.mp4 --relative --adapt_scale
