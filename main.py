#
# Copyright 2024 Sony Semiconductor Solutions Corp. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import atexit
import secrets
from picamera2 import Picamera2
from camera_manager import CameraManager
from flask import Flask, render_template, Response, abort, jsonify
from mobilenetv2 import Mobilenetv2_Annotator

# Global camera manager instance to keep track of cameras
camera_manager = CameraManager()

# Register an exit handler to clean up resources
@atexit.register
def app_exit():
    print("Exiting app.")
    camera_manager.release_all_cameras()

# Init Flask
app = Flask(__name__)
# Set Cookie settings
# Use a secure random secret key for session management
app.secret_key = secrets.token_hex(16)
# Set Same Site to Lax.
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# Set Logging from Picamera2
Picamera2.set_logging(Picamera2.DEBUG)

# Sets default routing for Home Page
@app.route('/')
def home():
    camera_list = camera_manager.refresh_camera_list()
    cameras = [camera.camera_num for camera in camera_list]
    return render_template('home.html', title="Raspberry Pi AI Camera Demo", camera_list=cameras, active_page='home')

# Sets routing to start video streaming and AI inference
@app.route('/start_video_stream_<int:camera_num>')
def start_video_stream(camera_num):

    print(f">> Starting Video Stream on camera {camera_num}")

    # Get IMX500 camera object based on camera number
    camera = camera_manager.get_imx500_camera_object(camera_num)

    if camera is None:
        print(f"Camera {camera_num} not found.")
        abort(404)

    else:
        # Use Mobile Net SSD model from "imx500-all" pacakge
        # To install the model on the Raspberry Pi, run: sudo apt install imx500-all
        imx500 = camera.initialize(ai_model='/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk')

        if imx500 is None:
            print(f"Failed to initialize camera {camera_num}.")
            abort(500)

        # Initialize Mobilenetv2_Annotator to process inference results
        # This is "Post Porocessing".  You may annotate the images with bounding boxes, labels, trigger events based on inference results, etc.
        annotator = Mobilenetv2_Annotator(camera)
        # The pre_callback, where the processing happens before the images are supplied to applications, before they arepassed to any video encoders, and before they are passed to any preview windows.
        camera.picamera2.pre_callback = annotator.pre_callback

        # Start video streaming & AI Inference
        camera.start_video_streaming()

        return Response(camera.stream_output.generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Sets routing to stop video streaming and AI inference
@app.route('/stop_video_stream_<int:camera_num>', methods=['GET'])
def stop_video_stream(camera_num):
    camera = camera_manager.get_imx500_camera_object(camera_num)
    try:
        if camera:
            camera.stop_video_streaming()
            response_data = {'result': 'success'}
            return jsonify(response_data)
        else:
            response_data = {'result': 'error', 'message': f'Camera {camera_num} not found.'}
            return jsonify(response_data)

    except Exception as e:
        return jsonify(success=False, error=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AiPiCam Demo')
    parser.add_argument('--port', type=int, default=8080, help='Web Server Port number')
    parser.add_argument('--ip', type=str, default='0.0.0.0', help='Web Server IP Address')
    args = parser.parse_args()
    print(f"Running Flask Server at {args.ip}:{args.port}")
    try:
        app.run(host=args.ip, port=args.port)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        print(f"Exception occurred: {e}")
