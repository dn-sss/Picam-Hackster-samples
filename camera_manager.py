import io
from threading import Condition
from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from picamera2.outputs import FileOutput
from picamera2.encoders import JpegEncoder
import time

# Manages a single IMX500 camera object.
class Imx500CameraObject:
    def __init__(self, camera_info):
        self.imx500 = None
        self.camera_info = camera_info
        self.camera_num = camera_info['Num'] if camera_info else None
        self.stream_output = None
        self.picamera2 = None

    # Initializes the IMX500 camera with the provided AI model.
    def initialize(self, ai_model):
        if self.camera_info:

            # Initialize the IMX500 camera with the provided AI model
            self.imx500 = IMX500(ai_model)

            # Check network intrinsics
            # Netowrk Intrinsics describes the network specific characteristics such as inference rate and labels.
            intrinsics = self.imx500.network_intrinsics

            # If no intrinsics are provided, create one for Object Detection
            if not intrinsics:
                intrinsics = NetworkIntrinsics()
                intrinsics.task = "object detection"

            # WPreserve image aspect ratio for input tensor image crop.
            if hasattr(intrinsics, 'preserve_aspect_ratio') and intrinsics.preserve_aspect_ratio:
                self.imx500.set_auto_aspect_ratio()

            # Update the network intrinsics with default values and above changes if any
            intrinsics.update_with_defaults()

            # Instantiate Picamera2 with the camera number
            self.picamera2 = Picamera2(self.camera_num)

            # Show progress bar in console while loading the network firmware
            self.imx500.show_network_fw_progress_bar()

            # return IMX500 camera object
            return self.imx500

        return None

    # Starts video streaming using Picamera2.
    def start_video_streaming(self):
        if self.picamera2 == None:
            print(f"Camera {self.camera_num} is not initialized.")
            return
        
        # if the camera is already started, stop it first
        if self.picamera2.started:
            self.stop_video_streaming()

        # Initialize stream output if not already done
        if self.stream_output is None:
            self.stream_output = StreamingOutput()

        # Start camera using Jpeg Encoder.
        # Output is written to the stream_output object.
        self.picamera2.start_recording(JpegEncoder(), output=FileOutput(self.stream_output))

    # Stops video streaming and releases resources.
    def stop_video_streaming(self):
        if self.picamera2 and self.picamera2.started:
            self.picamera2.stop_recording()
            self.picamera2.stop()
            self.stream_output = None
            self.picamera2.close()
            self.picamera2 = None
            del self.imx500
            self.imx500 = None
            print(f"Streaming on Camera {self.camera_num} stopped.")
        else:
            print(f"Streaming onCamera {self.camera_num} is not started or already stopped.")

# Manages cameras connected to the Raspberry Pi.
# Creates a list of "Imx500CameraObject) based on Picamera2's global camera info.

class CameraManager:
    def __init__(self):
        self.cameras = {}

    # Refreshes the camera list by querying Picamera2's global camera info
    def refresh_camera_list(self):
        # This method should return a list of camera info dictionaries
        camera_info = Picamera2.global_camera_info()
        self.cameras = {cam['Num']: Imx500CameraObject(cam) for cam in camera_info}
        return list(self.cameras.values())

    # returns Imx500CameraObject for the given camera number
    def get_imx500_camera_object(self, camera_num):
        return self.cameras.get(camera_num)
    
    def release_all_cameras(self):
        for camera in self.cameras.values():
            camera.stop_video_streaming()
        self.cameras.clear()

class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.buffer = io.BytesIO()
        self.condition = Condition()

    def write(self, buf):
        self.buffer.seek(0)
        self.buffer.truncate()
        self.buffer.write(buf)
        with self.condition:
            self.condition.notify_all()

    def read_frame(self):
        self.buffer.seek(0)
        return self.buffer.read()

    def generate_stream(self):
        while True:
            with self.condition:
                self.condition.wait()  # Wait for the new frame to be available
                frame = self.read_frame()
            if frame:
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

