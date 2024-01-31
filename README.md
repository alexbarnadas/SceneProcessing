# Scene Procerssing

Extrinsic transformation of the detections in an object detection application. With multiple cameras parallel processing. This project is made for the pilot use case in SERMAS for the ODIN project from the Horizon2020.

Assumptions: The camera is still and in a low abrupt angle.

## Installation
`python -m venv venv`

`pip install -r requirements.txt`
And you are ready to run main.py.

## Camera calibration
1. The first frame of the input video will apperar.
2. Click accurately in a squared area of the scene framed
3. Press esc

## Run Scene understanding
1. Specify the IP adresses of the cameras to be processed in real time in the main.py script.
2. Specify the conditions of the alerts in alerts.py.
3. Run main.py

The video will reproduce and also a window with the bird view coordinates of the detections will be drawn.

Every detection will be tracked and the orientation of the track will be shown with an arrow. 
Also the velocity in pixels per second will be plot frame by frame next to every detection
