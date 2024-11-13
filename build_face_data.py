from demo import load_checkpoints, make_animation, find_best_frame
from moviepy.editor import *
from skimage import img_as_ubyte
from skimage import io
from skimage.transform import resize
import cv2
import imageio
import os
import shutil
import yaml
from pathlib import Path

config = '/checkpoint/vox-256-spade.yaml'
checkpoint = '/checkpoint/00000189-checkpoint.pth.zip'

gen = 'spade'
cpu = False

generator, kp_detector, he_estimator = load_checkpoints(config_path=config, checkpoint_path=checkpoint, gen=gen, cpu=cpu)

def inference(source, driving, find_best_frame_ = False, free_view = False, yaw = None, pitch = None, roll = None, output_name = 'output.mp4',
              audio = True, cpu = False, best_frame = None, relative = True, adapt_scale = True,):
    # source 
    source_image = resize(source, (256, 256))
    
    # driving
    reader = imageio.get_reader(driving)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    
    with open(config) as f:
        config_ = yaml.load(f)
    estimate_jacobian = config_['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')

    if find_best_frame_ or best_frame is not None:
        i = best_frame if best_frame is not None else find_best_frame(source_image, driving_video, cpu=cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, he_estimator, relative=relative, adapt_movement_scale=adapt_scale, estimate_jacobian=estimate_jacobian, cpu=cpu, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
    
    # save video
    output_path = '/data'
    os.makedirs(output_path, exist_ok=True)
    
    print(f'{output_path}/{output_name}')    
    
    imageio.mimsave(f'{output_path}/{output_name}', [img_as_ubyte(frame) for frame in predictions], fps=fps)
    
    if audio:
        audioclip = VideoFileClip(driving)
        audio = audioclip.audio
        videoclip = VideoFileClip(f'{output_path}/{output_name}')
        videoclip.audio = audio
        name = output_name.strip('.mp4')
        videoclip.write_videofile(f'{output_path}/{name}_audio.mp4')
        return f'./{output_path}/{name}_audio.mp4'
    else:
        return f'{output_path}/{output_name}'


def build_face_data(source_image_path, driving_video_path):
    print("Start building face data")
    result = []
    for yaw in range(-30, 35, 5): # -30 to 30
        for pitch in range(-30, 35, 5):
            for roll in range(-30, 35, 5):
                output_name = Path(source_image_path).stem + '_' + Path(driving_video_path).stem + '_' + str(yaw) + '_' + str(pitch) + '_' + str(roll) +'.mp4'
                source = imageio.imread(source_image_path)
                output = inference(source, driving_video_path, yaw = yaw, pitch = pitch, roll = roll, output_name = output_name)
                result.append(output)

    print("Done building face data")
    print(result)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python build_face_data.py <source_image_path> <driving_video_path>")
        sys.exit(1)
    source_image_path = sys.argv[1]
    driving_video_path = sys.argv[2]
    #source_image_path = './uploads/1.png'
    #driving_video_path = './uploads/1.mp4'
    build_face_data(source_image_path, driving_video_path)
