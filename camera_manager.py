from picamera2 import Picamera2
from picamera2.devices.imx500 import IMX500, NetworkIntrinsics
from video_streaming import StreamingOutput
from picamera2.outputs import FileOutput
from picamera2.encoders import JpegEncoder
import time

# Camera management class for robust state handling
class Imx500CameraObject:
    def __init__(self, camera_info):
        self.imx500 = None
        self.camera_info = camera_info
        self.camera_num = camera_info['Num'] if camera_info else None
        self.stream_output = None
        self.picamera2 = None

    def initialize(self, ai_model):
        if self.camera_info:

            # Initialize the IMX500 camera with the provided AI model
            self.imx500 = IMX500(ai_model)

            # Check network intrinsics
            # Sample from Mobilenet v2
            #
            intrinsics = self.imx500.network_intrinsics

            if not intrinsics:
                intrinsics = NetworkIntrinsics()
                intrinsics.task = "object detection"

            if hasattr(intrinsics, 'preserve_aspect_ratio') and intrinsics.preserve_aspect_ratio:
                self.imx500.set_auto_aspect_ratio()

            intrinsics.update_with_defaults()

            self.picamera2 = Picamera2(self.camera_num)

            self.imx500.show_network_fw_progress_bar()

            return self.imx500

        return None

    def start_video_streaming(self):
        if self.picamera2:
            if self.picamera2.started:
                self.stop_video_streaming()
            
            self.imx500.show_network_fw_progress_bar()

            sensor_mode = self.picamera2.sensor_modes[0]
            inference_rate = min(10, self.imx500.network_intrinsics.inference_rate)
            self.video_config = self.picamera2.create_video_configuration(main={'size': sensor_mode['size']}, sensor={'output_size': sensor_mode['size']}, controls={'FrameRate': inference_rate}, buffer_count=12)
            self.picamera2.configure(self.video_config)

            if self.stream_output is None:
                self.stream_output = StreamingOutput()

            self.picamera2.start_recording(JpegEncoder(), output=FileOutput(self.stream_output))

    def stop_video_streaming(self):
        if self.picamera2 and self.picamera2.started:
            self.picamera2.stop()
            self.picamera2.close()
            self.picamera2 = None
            self.imx500 = None
            print(f"Camera {self.camera_num} stopped.")
        else:
            print(f"Camera {self.camera_num} is not started or already stopped.")

    def generate_stream(self):
        self.stream_output.generate_stream()

class CameraManager:
    def __init__(self):
        self.cameras = {}

    def refresh_camera_list(self):
        # This method should return a list of camera info dictionaries
        camera_info = Picamera2.global_camera_info()
        self.cameras = {cam['Num']: Imx500CameraObject(cam) for cam in camera_info}
        return list(self.cameras.values())

    def get_imx500_camera_object(self, camera_num):
        return self.cameras.get(camera_num)
