import os
import io
import sys
import time
import json
import argparse
import atexit
import secrets
import numpy as np
from camera_manager import CameraManager
from video_streaming import StreamingOutput
from flask import Flask, render_template, Response, request, abort, jsonify
from pprint import pprint, pformat
from picamera2 import Picamera2
from mobilenetv2 import Mobilenetv2_Annotater

# Global camera manager instance
camera_manager = CameraManager()

@atexit.register
def app_exit():
    print("Exiting app.")

# Init Flask
app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Set Logging from Picamera2
Picamera2.set_logging(Picamera2.DEBUG)

def generate_stream(camera):
    while True:
        with camera.stream_output.condition:
            camera.stream_output.condition.wait()  # Wait for the new frame to be available
            frame = camera.stream_output.read_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def home():
    camera_list = camera_manager.refresh_camera_list()
    cameras = [camera.camera_num for camera in camera_list]
    return render_template('home.html', title="Raspberry Pi AI Camera Demo", camera_list=cameras, active_page='home')

@app.route('/start_video_stream_<int:camera_num>')
def start_video_stream(camera_num):
    print(f">> Starting Video Stream on camera {camera_num}")

    camera = camera_manager.get_imx500_camera_object(camera_num)

    if camera is None:
        print(f"Camera {camera_num} not found.")
        abort(404)

    else:
        imx500 = camera.initialize(ai_model='/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk')

        if imx500 is None:
            print(f"Failed to initialize camera {camera_num}.")
            abort(500)

        camera.start_video_streaming()

        annotator = Mobilenetv2_Annotater(camera)
        camera.picamera2.pre_callback = annotator.pre_callback

        # return Response(generate_stream(camera), mimetype='multipart/x-mixed-replace; boundary=frame')
        return Response(camera.stream_output.generate_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_stream_<int:camera_num>', methods=['GET'])
def stop_video_stream(camera_num):
    print(f">> Stopping Video Stream on camera {camera_num}")
    camera = camera_manager.get_imx500_camera_object(camera_num)
    try:
        if camera:
            camera.stop_video_streaming()
        response_data = {'result': 'success'}
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