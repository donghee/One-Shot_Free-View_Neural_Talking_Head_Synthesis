from pathlib import Path
import gradio as gr
import os
import requests
import tempfile

samples = []

#example_source = os.listdir('asset/source')
example_source = os.listdir('/data')
for image in example_source:
    #samples.append([f'asset/source/{image}', None])
    samples.append([f'/data/{image}', None])

#example_driving = os.listdir('asset/driving')
example_driving = os.listdir('/data')
for video in example_driving:
    #samples.append([None, f'asset/driving/{video}'])
    samples.append([None, f'/data/{video}'])

API_URL = "http://127.0.0.1:8887/inference"

def inference(source, driving, find_best_frame = False, free_view = False, yaw = None, pitch = None, roll = None, output_name = 'output.mp4',
              audio = True, cpu = False, best_frame = None, relative = True, adapt_scale = True,):
    print("source: ", source, "driving: ", driving, "find_best_frame: ", find_best_frame, "free_view: ",
          free_view, "yaw: ", yaw, "pitch: ", pitch, "roll: ", roll, "output_name: ", output_name)
    r = requests.post(API_URL, 
                     files={"source_image": open(source, 'rb'), "driving_video": open(driving, 'rb')},
                     data={"find_best_frame": find_best_frame, "free_view": free_view, "yaw": yaw, "pitch": pitch, "roll": roll, "output_name": output_name})
    inferenced_video_url = r.json()["video_path"]
    inference_video_dir = tempfile.TemporaryDirectory().name
    if not os.path.exists(inference_video_dir):
        os.makedirs(inference_video_dir)
    inferenced_video = os.path.join(inference_video_dir, Path(inferenced_video_url).name)
    with open(inferenced_video, 'wb') as f:
        f.write(requests.get(inferenced_video_url).content)
    return inferenced_video

iface = gr.Interface(
    inference, # main function
    inputs = [ 
        #gr.Image(shape=(255, 255), label='Source Image'), # source image
        gr.Image(width=255, height=255, label='Source Image', type='filepath'), # source image
        #gr.Video(label='Driving Video', type='mp4'), # driving video
        gr.Video(label='Driving Video', format="mp4"), # driving video
        
        gr.Checkbox(label="fine best frame"),
        gr.Checkbox(label="free view"),
        gr.Slider(0, 90, value=0, label="yaw"),
        gr.Slider(0, 90, value=0, label="pitch"),
        gr.Slider(0, 90, value=0, label="roll"),
        #  gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),
    ],
    outputs = [
        gr.Video(label='result') # generated video
    ], 
    
    title = 'Face Vid2Vid Demo',
    description = "This app is an unofficial demo web app of the face video2video. The codes are heavily based on this repo, created by zhanglonghao1992",
    
    examples = samples)


iface.launch(share=True, server_name='0.0.0.0', server_port=8889)
