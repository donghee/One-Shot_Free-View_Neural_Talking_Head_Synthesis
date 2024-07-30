from demo import load_checkpoints, make_animation, find_best_frame
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from moviepy.editor import *
from skimage import img_as_ubyte
from skimage import io
from skimage.transform import resize
import cv2
import imageio
import os
import shutil
import yaml

app = FastAPI()

app.mount("/data", StaticFiles(directory="/data"), name="data")

@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OSFW</title>
    </head>
    <body>
        <h1>OSFW</h1>
        <form action="/inference" method="post" enctype="multipart/form-data">
            <label for="source_image">Source Image</label>
            <input type="file" id="source_image" name="source_image" accept="image/*" required><br><br>

            <label for="driving_video">Driving Video</label>
            <input type="file" id="driving_video" name="driving_video" accept="video/*" required><br><br>
            <label for="find_best_frame">Find Best Frame</label>
            <input type="checkbox" id="find_best_frame" name="find_best_frame" value="true"><br><br>
            <label for="free_view">Free View</label>
            <input type="checkbox" id="free_view" name="free_view" value="true"><br><br>
            <label for="yaw">Yaw</label>
            <input type="number" id="yaw" name="yaw" value="0"><br><br>
            <label for="pitch">Pitch</label>
            <input type="number" id="pitch" name="pitch" value="0"><br><br>
            <label for="roll">Roll</label>
            <input type="number" id="roll" name="roll" value="0"><br><br>
            <label for="output_name">Output Name</label>
            <input type="text" id="output_name" name="output_name" value="output.mp4"><br><br>
            <br>

            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

config = '/checkpoint/vox-256-spade.yaml'
checkpoint = '/checkpoint/00000189-checkpoint.pth.zip'

gen = 'spade'
cpu = False

generator, kp_detector, he_estimator = load_checkpoints(config_path=config, checkpoint_path=checkpoint, gen=gen, cpu=cpu)

def inference(source, driving, find_best_frame = False, free_view = False, yaw = None, pitch = None, roll = None, output_name = 'output.mp4',
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

    if find_best_frame or best_frame is not None:
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
    #output_path = 'asset/output'
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

@app.post("/inference")
async def post_inference(source_image: UploadFile = File(...), driving_video: UploadFile = File(...),
                    find_best_frame: bool = Form(...), free_view: bool = Form(...), yaw: float = Form(...),
                    pitch: float = Form(...), roll: float = Form(...), output_name: str = Form(...)

                    ):

    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    source_image_path = os.path.join('uploads', source_image.filename)
    driving_video_path = os.path.join('uploads', driving_video.filename)

    with open(source_image_path, 'wb') as f:
        shutil.copyfileobj(source_image.file, f)

    with open(driving_video_path, 'wb') as f:
        shutil.copyfileobj(driving_video.file, f)

    print(source_image_path, driving_video_path, find_best_frame, free_view, yaw, pitch, roll, output_name)

    source = imageio.imread(source_image_path)
    target_video_path = inference(source, driving_video_path, find_best_frame = False, free_view =
                                  False, yaw = None, pitch = None, roll = None, output_name = 'output.mp4')

    target_video_path = '/data/' + output_name
    print(target_video_path)
 
    return JSONResponse(content={'video_path': 'http://127.0.0.1:8887' + target_video_path})

if __name__ == '__main__':
    import uvicorn
    #uvicorn.run(app, host='0.0.0.0', port=8888)
    uvicorn.run(app, host='0.0.0.0', port=8887)
