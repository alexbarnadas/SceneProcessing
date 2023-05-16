import numpy as np
import cv2

from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator

from Calibrator import SceneCalibration

source = './Terraza.avi'

cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)
intra_frame_delay = 1 #int(1000/fps)#ms

# Scene calibration
rval, first_frame = cap.read()
first_frame = cv2.resize(first_frame, (1280,720), interpolation = cv2.INTER_LINEAR)

Calibration = SceneCalibration(first_frame.copy())
perspective_matrix = Calibration.perspective_matrix
print(perspective_matrix)


bird_view = cv2.warpPerspective(first_frame, perspective_matrix, (600,600))
(threshold, bird_map) = cv2.threshold(bird_view, 0, 200, cv2.THRESH_BINARY)
#cv2.imshow('Bird View', bird_view)
#cv2.imshow('Bird Map',  bird_map)

# Object detection
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
while (cap.isOpened()):
  ret, frame = cap.read()
  if ret == True:
    frame = cv2.resize(frame, (1280,720), interpolation = cv2.INTER_LINEAR)

    cv2.line(frame, Calibration.points[0], Calibration.points[1], (255,0,255), 2)
    cv2.line(frame, Calibration.points[0], Calibration.points[2], (255,0,255), 2)
    cv2.line(frame, Calibration.points[2], Calibration.points[3], (255,0,255), 2)
    cv2.line(frame, Calibration.points[3], Calibration.points[1], (255,0,255), 2)

    results = model.predict(frame, classes=[0,15])  # predict on an image

    for r in results:
      boxes = r.boxes
      for box in boxes:     
          b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
          b_xc = int((b[0] + b[2])/2)
          b_y =  int(b[3])
          cv2.circle(frame, (b_xc, b_y), 4,(0,0,255),2)
          print('coords: ', np.array([[b_xc, b_y, 1]]))
          bird_xcys = np.matmul(perspective_matrix, np.array([b_xc, b_y, 1]))
          bird_coords = (int(bird_xcys[0]/bird_xcys[2]), int(bird_xcys[1]/bird_xcys[2]))
          print('coords: ', bird_coords)
          bird_map = cv2.circle(bird_map, bird_coords, 4, (0,255,0), 2)
    
    # Uncomment to see the unwrapped image
    #frame = cv2.warpPerspective(frame, perspective_matrix, (600,800))     cv2.imshow('Video', frame)
    cv2.imshow('Map', bird_map)
    cv2.imshow('Camera view', frame)

    if cv2.waitKey(intra_frame_delay) & 0xFF == ord('q'): break
  else: break
cap.release()
cv2.destroyAllWindows()