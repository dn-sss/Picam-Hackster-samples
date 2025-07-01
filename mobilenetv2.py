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

# Inspired by the Picamera2 examples, this code annotates the MobileNet SSD v2 inference results
# Based on https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py

from time import time
import cv2
import numpy as np
from functools import lru_cache
from picamera2 import MappedArray
from picamera2.devices.imx500 import (NetworkIntrinsics,
                                      postprocess_nanodet_detection)

class Detection:
    def __init__(self, coords, category, conf, metadata, imx500, picam2):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

# A class to annotate Mobile Net SSD v2 inference results.
# Taken codes from the Picamera2 examples.

class Mobilenetv2_Annotater:
    def __init__(self, imx500_camera_object, detection_threashold=0.55, iou=0.65, max_detections=10):
        self.imx500_camera_object = imx500_camera_object
        self.imx500 = imx500_camera_object.imx500
        self.detection_threashold = detection_threashold
        self.iou = iou
        self.max_detections = max_detections
        self.last_results = None
        self.last_frame_time = 0

    # A callback function from Picamera2 that is called before the image is processed by the video encoders or preview windows.
    def pre_callback(self, request):
        try:
            fps = 0
            new_frame_time = time()
            if (self.last_frame_time != 0):
                fps = 1 / (new_frame_time - self.last_frame_time)
            self.last_frame_time = new_frame_time
            metadata = request.get_metadata()
            self.last_results = self.parse_metadata(metadata)
            self.draw_detections(request, fps)
        except Exception as e:
            print(f"Exception in pre_callback() : {e}")

    def parse_metadata(self, metadata: dict):
        intrinsics = self.imx500.network_intrinsics
        bbox_normalization = intrinsics.bbox_normalization
        bbox_order = intrinsics.bbox_order
        threshold = self.detection_threashold
        iou = self.iou
        max_detections = self.max_detections

        # Gets output tensor from IMX500.
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)

        # Gets Model Input Tensor Size
        input_w, input_h = self.imx500.get_input_size()

        # If no outputs, return previous results so that the screen won't flicker.
        if np_outputs is None:
            return self.last_results

        # from https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py
        boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
        if bbox_normalization:
            boxes = boxes / input_h

        if bbox_order == "xy":
            boxes = boxes[:, [1, 0, 3, 2]]
        boxes = np.array_split(boxes, 4, axis=1)
        boxes = zip(*boxes)

        results = [
            Detection(box, category, score, metadata, self.imx500, self.imx500_camera_object.picamera2)
            for box, score, category in zip(boxes, scores, classes)
            if score > threshold
        ]
        return results

    def draw_detections(self, request, fps):
        """Draw the detections for this request onto the ISP output."""
        detections = self.last_results
        if detections is None:
            return

        labels = self.get_labels()
        intrinsics = self.imx500.network_intrinsics
        image_w = self.imx500_camera_object.picamera2.video_configuration.main.size[0]
        image_h = self.imx500_camera_object.picamera2.video_configuration.main.size[1]
        line_thickness = round(6 * (image_w / 2048))
        font_scale = round(1 * (image_w / 2048))

        with MappedArray(request, 'main') as m:
            for detection in detections:
                x, y, w, h = detection.box
                label = f"{labels[int(detection.category)]} ({(detection.conf * 100):.1f}%)"

                # Calculate text size and position
                (text_w, text_h), text_base = cv2.getTextSize(text=label,
                                                              fontFace=cv2.FONT_HERSHEY_DUPLEX  ,
                                                              fontScale=font_scale,
                                                              thickness=1)
                margin = round(text_h * 0.1)
                text_x = x + line_thickness + margin
                text_y = y + line_thickness + margin + text_h

                # Create a copy of the array to draw the background with opacity
                overlay = m.array.copy()
                # Draw the background rectangle on the overlay
                rect_x = x + round(line_thickness/2)
                rect_y = y + round(line_thickness/2)
                rect_w = text_w + (2 * margin)
                rect_h = text_h + (2 * margin) + text_base
                cv2.rectangle(overlay,
                            (rect_x, rect_y),
                            ((rect_x + rect_w), (rect_y + rect_h)),
                            (255, 255, 255), 
                            cv2.FILLED)

                alpha = 0.30
                cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)


                # Draw text on top of the background
                cv2.putText(m.array,
                            text=label,
                            org=(text_x, text_y),
                            fontFace=cv2.FONT_HERSHEY_DUPLEX  ,
                            fontScale=font_scale,
                            color=(0, 0, 0),
                            thickness=1)

                # Draw detection box
                cv2.rectangle(m.array,
                              pt1=(x, y),
                              pt2=(x + w, y + h),
                              color=(0, 255, 0),
                              thickness=line_thickness)

            fps_text = f"FPS: {int(fps)}"

            (text_w, text_h), text_base = cv2.getTextSize(text=fps_text,
                                                          fontFace=cv2.FONT_HERSHEY_DUPLEX  ,
                                                          fontScale=font_scale,
                                                          thickness=1)
            margin = round(text_h * 0.1)
            text_x = image_w - text_w - margin
            text_y = image_h - text_h - margin

            # Draw FPS text on the right bottom corner
            cv2.putText(m.array,
                        text=fps_text,
                        org=(text_x, text_y),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX  ,
                        fontScale=font_scale,
                        color=(0, 0, 0),
                        thickness=1)

    @lru_cache
    def get_labels(self):
        intrinsics = self.imx500.network_intrinsics
        labels = intrinsics.labels

        if intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels
