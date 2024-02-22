from collections import defaultdict
import numpy as np
import math
import yaml
import csv
import pandas as pd
import cv2
from ultralytics import YOLO

from calibrator import SceneCalibration

# Init the track history
track_history = defaultdict(lambda: [])
track_history_warped = defaultdict(lambda: [])

# Options
SHOW = True
SAVE = False
CALLIBRATE = True
UNDISTORT = True
dt = 20  # frames
history_save_interval = (30*60)  # Save every minute (1800 frames)

# Load the YOLov8 model
model = YOLO('Models/yolov8n-pose.pt')  # load a pretrained model (recommended for training)

source = 'TestVideos/camkitchen_rec2.mp4'

cap = cv2.VideoCapture(source)
frame_size, fps = (int(cap.get(3)), int(cap.get(4))), cap.get(5)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('Demo.mp4', fourcc, fps, frame_size)
writer_map = cv2.VideoWriter('Demo_map.mp4', fourcc, fps, (600, 600))
# Load the map
#  bird_map = cv2.imread('LivingLab_2D.png')
#  bird_map = cv2.resize(bird_map, (600, 600))

# Load fisheye camera calibration variables
with open('calibration_camkitchen_results_good.yaml', 'r') as file:
    fisheye_correction = yaml.load(file, Loader=yaml.FullLoader)

mtx = np.array(fisheye_correction['PARAMETERS']['INTRINSIC']['calibration_matrix'])
dist = np.array(fisheye_correction['PARAMETERS']['INTRINSIC']['distortion_coefficients'][0])

# Scene and camera calibration
success, first_frame = cap.read()
h, w = first_frame.shape[:2]
cv2.imwrite('first_frame.jpg', first_frame)
if not success:
    print('Error while loading calibration frame')
    exit()

new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted_frame = cv2.undistort(first_frame, mtx, dist, None, new_camera_mtx)
x, y, w, h = roi

undistorted_frame = undistorted_frame[y:y+h, x:x+w]

cv2.imshow('Undistorted Frame', undistorted_frame)

Calibration = SceneCalibration(undistorted_frame.copy())
perspective_matrix = Calibration.perspective_matrix
print(perspective_matrix)

bird_view = cv2.warpPerspective(first_frame, perspective_matrix, (600, 600))
(threshold, bird_map) = cv2.threshold(bird_view, 0, 200, cv2.THRESH_BINARY)
# cv2.imshow('Bird View', bird_view)

# Output result for Angelo
people_flow_history = []

#  Write result into a file
header = ['FrameNumber', 'ScreenCoords', 'SceneCoords', 'Velocity', 'Orientation']
csvfile = open('PeopleFlowHistoryPD.csv', 'w')
c = csv.DictWriter(csvfile, header)
c.writeheader()
csvfile.close()

frame_id = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)

    cv2.line(frame, Calibration.points[0], Calibration.points[1], (255, 0, 255), 2)
    cv2.line(frame, Calibration.points[0], Calibration.points[2], (255, 0, 255), 2)
    cv2.line(frame, Calibration.points[2], Calibration.points[3], (255, 0, 255), 2)
    cv2.line(frame, Calibration.points[3], Calibration.points[1], (255, 0, 255), 2)

    results = model.track(frame, persist=True)

    # Get the boxes and track IDs
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.tolist()
        keypoints = results[0].keypoints.xy
    else: continue

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Plot the tracks
    bird_map_written = bird_map.copy()
    for box, track_id, cls, keypt in zip(boxes, track_ids, classes, keypoints):
        if cls != 0.0: continue

        x_box, y_box, w, h = box
        y_box += h / 2

        if keypt[15][0] != 0 and keypt[16][0] != 0:
            feet = (keypt[15] + keypt[16]).div(2).cpu()
            x, y = feet
        elif keypt[6][0] != 0 and keypt[5][0] != 0 and False:
            heart = (keypt[6] + keypt[5]).div(2).cpu()
            x, y = heart
            shoulder = np.linalg.norm((keypt[6] - keypt[5]).cpu())
            y += shoulder * 8
            cv2.circle(annotated_frame, (int(x), int(y)), 40, (255, 0, 255), 20)
        else: continue

        # Bird coordinates transformation
        wx, wy, w_scale = np.matmul(perspective_matrix, np.array([x, y, 1]))
        bird_coords = [int(wx / w_scale), int(wy / w_scale)]
        # print('Coords:', [int(x), int(y)], '->', bird_coords)
        # bird_map = cv2.circle(bird_map, bird_coords, 4, (0, 255, 0), 2)

        x1, y1 = bird_coords

        track = track_history[track_id]
        track_warped = track_history_warped[track_id]

        if len(track) > dt:  # Calculate track velocity
            x2, y2 = track_warped[-dt][0], track_warped[-dt][1]
            dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
            velocity = int(dist * fps / dt)  # Velocity assuming fps is consistent with real time
            orientation = int(math.degrees(math.atan2(y2 - y1, x2 - x1)))  # Orientation in rad
            bird_map_written = cv2.arrowedLine(bird_map_written,
                                               (int(x1), int(y1)),
                                               (int(x2 - (x2 - x1) * dt / 15), int(y2 - (y2 - y1) * dt / 15)),
                                               (0, 0, 255), 3)
            bird_map_written = cv2.putText(bird_map_written, '' +
                                           'Vel' + str(velocity) + 'pix/s',
                                           # 'Dir'+str(orientation),
                                           (int(x1 + w / 4), int(y1)),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (130, 130, 130), 2)
        else: velocity = orientation = None

        track.append((float(x), float(y)))  # x, y center-ground point
        track_warped.append(bird_coords)

        if len(track) > 30:
            track.pop(0)
            track_warped.pop(0)

        # Draw the tracking lines
        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

        points_warped = np.hstack(track_warped).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(bird_map_written, [points_warped], isClosed=False, color=(230, 230, 230), thickness=3)

        people_flow_history.append({
            'FrameNumber':   frame_id,        # id of the current frame
            'ScreenCoords':  box.int().tolist(),  # x,y,h,w
            'SceneCoords':   bird_coords,   # real_x1,real_x2
            'Velocity':      velocity,      # pix/s
            'Orientation':   orientation    # radians
        })
    frame_id += 1

    if len(people_flow_history) % history_save_interval == 0:
        people_flow_df = pd.DataFrame(people_flow_history)
        people_flow_df.to_csv('PeopleFlowHistoryPD.csv', mode='a', header=False, index=False)
        people_flow_history = []

    if SAVE:
        writer.write(annotated_frame)
        writer_map.write(bird_map_written)

    # Uncomment to see the unwrapped image
    if SHOW:
        # frame_warped = cv2.warpPerspective(frame, perspective_matrix, (600,800))
        # Uncomment to see bird perspective image
        # cv2.imshow('Video', frame_warped)
        cv2.imshow('Camera view', annotated_frame)
        cv2.imshow('Map', bird_map_written)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
writer.release()
cv2.destroyAllWindows()
