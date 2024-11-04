from pathlib import Path
import gradio as gr
import os
import requests
import tempfile

import numpy as np
import base64
import io
from PIL import Image
from datetime import datetime

samples = []

example_source = os.listdir('/data')
for image in example_source:
    samples.append([f'/data/{image}', None])

example_driving = os.listdir('/data')
for video in example_driving:
    samples.append([None, f'/data/{video}'])

API_URL = "http://127.0.0.1:8887/inference"


def check_downloadable(url):
    try:
        response = requests.head(url, allow_redirects=True)

        # Check if the response status code is OK (200)
        if response.status_code == 200:
            # Check if the 'Content-Type' in the header indicates a file
            content_type = response.headers.get('Content-Type', '')
            if 'application' in content_type or 'octet-stream' in content_type or 'video' in content_type:
                return True
            else:
                return False
        else:
            return False
    except requests.RequestException as e:
        print(f"Error: {e}")
        return False

def inference_video(source, driving, find_best_frame, free_view, yaw, pitch, roll):
    output_name = 'output.mp4'
    output_name = Path(source).stem + '_' + Path(driving).stem + '_' + str(yaw) + '_' + str(pitch) + '_' + str(roll) +'.mp4'

    print("source: ", source, "driving: ", driving, "find_best_frame: ", find_best_frame, "free_view: ",
          free_view, "yaw: ", yaw, "pitch: ", pitch, "roll: ", roll, "output_name: ", output_name)

    # check can download url
    inferenced_video_url = "http://127.0.0.1:8887/data/"+output_name
    if check_downloadable(inferenced_video_url):
        print('has output')
    else:
        r = requests.post(API_URL,
                          files={"source_image": open(source, 'rb'), "driving_video": open(driving, 'rb')},
                          data={"find_best_frame": find_best_frame, "free_view": free_view, "yaw": yaw, "pitch": pitch, "roll": roll, "output_name": output_name})
        inferenced_video_url = r.json()["video_path"]
    return inferenced_video_url, inferenced_video_url
    # inference_video_dir = tempfile.TemporaryDirectory().name
    # if not os.path.exists(inference_video_dir):
    #     os.makedirs(inference_video_dir)
    # inferenced_video = os.path.join(inference_video_dir, Path(inferenced_video_url).name)
    # with open(inferenced_video, 'wb') as f:
    #     f.write(requests.get(inferenced_video_url).content)
    # return inferenced_video, inferenced_video

def key_changed(yaw, pitch, roll, key):
    if key == "q":
        yaw -= 5
    elif key == "e":
        yaw += 5
    elif key == "w":
        pitch += 5
    elif key == "s":
        pitch -= 5
    elif key == "a":
        roll -= 5
    elif key == "d":
        roll += 5
    return [yaw, pitch, roll, yaw, pitch, roll]

def save_video_frame(ai_avatar_video_frame_data, yaw_exp, pitch_exp, roll_exp, user_video_1_frame_data, user_video_2_frame_data, user_video_3_frame_data):
    if ai_avatar_video_frame_data is None or ai_avatar_video_frame_data == "":
        return None

    ai_avatar_img_data = base64.b64decode(ai_avatar_video_frame_data.split(',')[1])
    ai_avatar_img = Image.open(io.BytesIO(ai_avatar_img_data))

    if user_video_1_frame_data is None or user_video_1_frame_data == "":
        user_video_1_img = Image.new("RGB", ai_avatar_img.size, (255, 255, 255))
    else:
        user_video_1_img_data = base64.b64decode(user_video_1_frame_data.split(',')[1])
        user_video_1_img = Image.open(io.BytesIO(user_video_1_img_data))

    if user_video_2_frame_data is None or user_video_2_frame_data == "":
        user_video_2_img = Image.new("RGB", ai_avatar_img.size, (255, 255, 255))
    else:
        user_video_2_img_data = base64.b64decode(user_video_2_frame_data.split(',')[1])
        user_video_2_img = Image.open(io.BytesIO(user_video_2_img_data))

    if user_video_3_frame_data is None or user_video_3_frame_data == "":
        user_video_3_img = Image.new("RGB", ai_avatar_img.size, (255, 255, 255))
    else:
        user_video_3_img_data = base64.b64decode(user_video_3_frame_data.split(',')[1])
        user_video_3_img = Image.open(io.BytesIO(user_video_3_img_data))

    # save Train_data
    os.makedirs('/Train_data', exist_ok=True)
    now = datetime.now().strftime("%y%m%d%H%M%S")
    csv_file = open(f'/Train_data/rotation.csv', 'a')
    csv_file.write(f'{now}, {roll_exp}, {pitch_exp}, {yaw_exp}\n')
    csv_file.close()
    ai_avatar_img.save(f'/Train_data/{now}_av_img0001.jpg')
    user_video_1_img.save(f'/Train_data/{now}_cam1_img0001.jpg')
    user_video_2_img.save(f'/Train_data/{now}_cam2_img0001.jpg')
    user_video_3_img.save(f'/Train_data/{now}_cam3_img0001.jpg')

    return [np.array(ai_avatar_img), np.array(user_video_1_img), np.array(user_video_2_img), np.array(user_video_3_img)]

def get_latest_video(dir, format):
    sorted_video_list_by_time = sorted(os.listdir(dir), key=lambda x: os.path.getmtime(os.path.join(dir, x)), reverse=True)
    return dir + '/' + sorted_video_list_by_time[0]

css = """
#warning {background-color: #FFCCCB}
.feedback textarea {font-size: 24px !important}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as iface:
    gr.Markdown("# AI Avatar")
    with gr.Tab(label="OSFW") as osfw_tab:
        gr.Markdown("## OSFW")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input")

                source = gr.Image(label='Source Image', type='filepath')
                driving = gr.Video(label='Driving Video', format="mp4")

                find_best_frame = gr.Checkbox(label="fine best frame")
                free_view = gr.Checkbox(label="free view", value=True)
                roll = gr.Slider(-30, 30, value=0, label="roll")
                pitch = gr.Slider(-30, 30, value=0, label="pitch")
                yaw = gr.Slider(-30, 30, value=0, label="yaw")
                key_press = gr.Textbox(visible=False)

                submit = gr.Button(value="Submit")
            with gr.Column():
                gr.Markdown("### Result")
                video_output = gr.Video(label='Result Video', elem_id="result_video", autoplay=True, interactive=False, container=False) # generated video

    with gr.Tab(label="EXP") as exp_tab:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## AI 아바타 영상")
                ai_avatar_video = gr.Video(label='AI Avatar Video', elem_id="ai_avatar_video", autoplay=True, height=640)
                ai_avatar_video_frame = gr.Image(label='AI Avatar Video Frame', visible=False)
                ai_avatar_video_frame_data = gr.Textbox(visible=False)

                roll_exp = gr.Slider(-30, 30, value=0, label="roll")
                pitch_exp = gr.Slider(-30, 30, value=0, label="pitch")
                yaw_exp = gr.Slider(-30, 30, value=0, label="yaw")

                save = gr.Button(value="Save")

                gr.Markdown("## 사용자 시선영상")
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            user_video_1 = gr.Video(label='User Video 1', elem_id="user_video_1")
                            user_video_2 = gr.Video(label='User Video 2', elem_id="user_video_2")
                            user_video_3 = gr.Video(label='User Video 3', elem_id="user_video_3")

                        with gr.Row():
                            user_video_1_frame = gr.Image(label='User Video 1 Frame', visible=False, sources=['webcam'])
                            user_video_2_frame = gr.Image(label='User Video 2 Frame', visible=False)
                            user_video_3_frame = gr.Image(label='User Video 3 Frame', visible=False)
                        user_video_1_frame_data = gr.Textbox(visible=False)
                        user_video_2_frame_data = gr.Textbox(visible=False)
                        user_video_3_frame_data = gr.Textbox(visible=False)

            # for keyboard control
            key_press.change(key_changed, inputs=[yaw, pitch, roll, key_press], outputs=[yaw, pitch, roll, yaw_exp, pitch_exp, roll_exp], show_progress=False)
            yaw_exp.change(inference_video, [source, driving, find_best_frame, free_view, yaw, pitch, roll], [video_output, ai_avatar_video], show_progress=False)
            pitch_exp.change(inference_video, [source, driving, find_best_frame, free_view, yaw, pitch, roll], [video_output, ai_avatar_video], show_progress=False)
            roll_exp.change(inference_video, [source, driving, find_best_frame, free_view, yaw, pitch, roll], [video_output, ai_avatar_video], show_progress=False)
            submit.click(inference_video, [source, driving, find_best_frame, free_view, yaw, pitch, roll], [video_output, ai_avatar_video], show_progress=False)

            # save data for experiment
            save.click(save_video_frame, [ai_avatar_video_frame_data, yaw_exp, pitch_exp, roll_exp, user_video_1_frame_data, user_video_2_frame_data, user_video_3_frame_data], [ai_avatar_video_frame, user_video_1_frame, user_video_2_frame, user_video_3_frame],
                       js="""
                       function(ai_avatar_video_frame_data, yaw_exp, pitch_exp, roll_exp, user_video_1_frame_data, user_video_2_frame_data, user_video_3_frame_data) {
                           var video = document.querySelector('#ai_avatar_video video');
                           var canvas = document.createElement('canvas');
                           canvas.width = video.videoWidth;
                           canvas.height = video.videoHeight;
                           canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                           ai_avatar_video_frame_data = canvas.toDataURL('image/jpeg');

                           video = document.querySelector('#user_video_1 video');
                           if (video != null) {
                           canvas.width = video.videoWidth;
                           canvas.height = video.videoHeight;
                           canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                           user_video_1_frame_data = canvas.toDataURL('image/jpeg');
                           }

                           video = document.querySelector('#user_video_2 video');
                           if (video != null) {
                           canvas.width = video.videoWidth;
                           canvas.height = video.videoHeight;
                           canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                           user_video_2_frame_data = canvas.toDataURL('image/jpeg');
                           }

                           video = document.querySelector('#user_video_3 video');
                           if (video != null) {
                           canvas.width = video.videoWidth;
                           canvas.height = video.videoHeight;
                           canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
                           user_video_3_frame_data = canvas.toDataURL('image/jpeg');
                           }

                           return [ai_avatar_video_frame_data, yaw_exp, pitch_exp, roll_exp, user_video_1_frame_data, user_video_2_frame_data, user_video_3_frame_data];
                      }
                       """
                       )


        def on_exp_tab_update(video_output, yaw, pitch, roll):
            return video_output, yaw, pitch, roll

    exp_tab.select(on_exp_tab_update, [video_output, yaw, pitch, roll], [ai_avatar_video, yaw_exp, pitch_exp, roll_exp])

    with gr.Tab(label="P.E.") as pe_tab:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## AI 아바타 영상")
                pe_video = get_latest_video('/Test_data', 'mp4')
                ai_avatar_video_pe = gr.Video(label=f'AI Avatar Video: {pe_video}', format="mp4",
                                              elem_id="ai_avatar_video_pe", autoplay=True, height=640, 
                                              loop=True, value=pe_video)

                gr.Markdown("## 사용자 시선영상")
                with gr.Row():
                    user_video_1_pe = gr.Video(label='User Video')
                    user_video_2_pe = gr.Video(label='User Video')
                    user_video_3_pe = gr.Video(label='User Video')

        def on_pe_tab_update(video_output):
            return video_output

    #pe_tab.select(on_pe_tab_update, [video_output], [ai_avatar_video_pe])

    iface.load(fn=None, inputs=None, outputs=None,
        js="""
        function() {
            document.addEventListener('keydown', function(e) {
                document.querySelector('textarea[data-testid="textbox"]').value = ''; // clear the text box
                document.querySelector('textarea[data-testid="textbox"]').dispatchEvent(new Event('input'));
                document.querySelector('textarea[data-testid="textbox"]').value = e.key;
                document.querySelector('textarea[data-testid="textbox"]').dispatchEvent(new Event('input'));
            });
        }
        """)

iface.launch(share=True, server_name='0.0.0.0', server_port=8889)
