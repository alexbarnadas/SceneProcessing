# YOLOv8_Extrinsic
A proof of concept of an Extrinsic transformation of the detections in an object detection application.

Assumptions: The camera is still and in a low abrupt angle.

Instructions:
1. Run yolov8 detector, the first frame of the input video will apperar.
2. Click accurately in a squared area of the scene framed
3. Press esc

The video will reproduce and also a window with the bird view coordinates of the detections will be drawn.

Every detection will be tracked and the orientation of the track will be shown with an arrow. 
Also the velocity in pixels per second will be plot frame by frame next to every detection