#!/bin/sh

python /One-Shot_Free-View_Neural_Talking_Head_Synthesis/demo.py --result_video /data/result.mp4 --config /checkpoint/vox-256-spade.yaml --checkpoint /checkpoint/00000189-checkpoint.pth.zip --source_image /data/man.png --driving_video /data/man.mp4 --relative --adapt_scale 
#python demo.py --result_video /data/result.mp4 --config /checkpoint/vox-256-spade.yaml --checkpoint /checkpoint/00000189-checkpoint.pth.zip --source_image /data/man.png --driving_video /data/man.mp4 --relative --adapt_scale  --find_best_frame
