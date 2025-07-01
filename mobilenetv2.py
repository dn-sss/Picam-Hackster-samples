# Inspired by the Picamera2 examples, this code annotates the MobileNet SSD v2 inference results
# Based on https://github.com/raspberrypi/picamera2/blob/main/examples/imx500/imx500_object_detection_demo.py

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

    # A callback function from Picamera2 that is called before the image is processed by the video encoders or preview windows.
    def pre_callback(self, request):
        try:
            metadata = request.get_metadata()
            self.last_results = self.parse_metadata(metadata)
            self.draw_detections(request)
        except Exception as e:
            print(f"Exception : {e}")
            breakpoint()

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

    def draw_detections(self, request):
        """Draw the detections for this request onto the ISP output."""
        detections = self.last_results
        if detections is None:
            return

        labels = self.get_labels()
        intrinsics = self.imx500.network_intrinsics
        image_w = self.imx500_camera_object.picamera2.video_configuration.main.size[0]
        line_thickness = round(6 * (image_w / 2048))
        font_scale = round(1 * (image_w / 2048))

        with MappedArray(request, 'main') as m:
            for detection in detections:
                x, y, w, h = detection.box
                label = f"{labels[int(detection.category)]} ({(detection.conf * 100):.1f}%)"

                # Calculate text size and position
                (text_w, text_h), text_base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
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
                cv2.putText(m.array, label, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 1)

                # Draw detection box
                cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=line_thickness)

            if intrinsics.preserve_aspect_ratio:
                b_x, b_y, b_w, b_h = self.imx500.get_roi_scaled(request)
                color = (255, 0, 0)  # red
                cv2.putText(m.array, "ROI", (b_x + 5, b_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                cv2.rectangle(m.array, (b_x, b_y), (b_x + b_w, b_y + b_h), (255, 0, 0, 0))

    @lru_cache
    def get_labels(self):
        intrinsics = self.imx500.network_intrinsics
        labels = intrinsics.labels

        if intrinsics.ignore_dash_labels:
            labels = [label for label in labels if label and label != "-"]
        return labels
