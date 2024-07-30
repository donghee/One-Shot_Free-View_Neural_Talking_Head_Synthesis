import os
import gradio as gr
   
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

def inference(source,
              driving,
              find_best_frame_ = False,
              free_view = False,
              yaw = None,
              pitch = None,
              roll = None,
              output_name = 'output.mp4',
              
              audio = True,
              cpu = False,
              best_frame = None,
              relative = True,
              adapt_scale = True,
              ):
    pass

iface = gr.Interface(
    inference, # main function
    inputs = [ 
        #gr.Image(shape=(255, 255), label='Source Image'), # source image
        gr.Image(label='Source Image'), # source image
        #gr.Video(label='Driving Video', type='mp4'), # driving video
        gr.Video(label='Driving Video'), # driving video
        
        gr.Checkbox(label="fine best frame"),
        gr.Checkbox(label="free view"),
        gr.Slider(0, 90, value=0, label="yaw"),
        gr.Slider(0, 90, value=0, label="pitch"),
        gr.Slider(0, 90, value=0, label="raw"),
               gr.Slider(2, 20, value=4, label="Count", info="Choose between 2 and 20"),
        
    ],
    outputs = [
        gr.Video(label='result') # generated video
    ], 
    
    title = 'Face Vid2Vid Demo',
    description = "This app is an unofficial demo web app of the face video2video. The codes are heavily based on this repo, created by zhanglonghao1992",
    
    examples = samples)


iface.launch(share=True, server_name='0.0.0.0', server_port=8889)
