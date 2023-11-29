from collections import defaultdict
import numpy as np
import math
import cv2
from ultralytics import YOLO

from Calibrator import SceneCalibration

# Options
SHOW = True

# Load the YOLov8 model
model = YOLO("yolov8n-pose.pt")  # load a pretrained model (recommended for training)

source = "Videos_UPM/camlivingroom_rec1.mp4"
cap = cv2.VideoCapture(source)
width, height, fps = cap.get(3), cap.get(4), cap.get(5)
fourcc = cv2.VideoWriter_fourcc(*'avc1')
writer = cv2.VideoWriter('Demo.mp4', fourcc, fps, (1280, 720))

# Store the track history
track_history = defaultdict(lambda: [])
track_history_warped = defaultdict(lambda: [])

# Scene calibration
_, first_frame = cap.read()
first_frame = cv2.resize(first_frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

Calibration = SceneCalibration(first_frame.copy())
perspective_matrix = Calibration.perspective_matrix
print(perspective_matrix)

bird_view = cv2.warpPerspective(first_frame, perspective_matrix, (600, 600))
(threshold, bird_map) = cv2.threshold(bird_view, 0, 200, cv2.THRESH_BINARY)
# cv2.imshow('Bird View', bird_view)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        cv2.line(frame, Calibration.points[0], Calibration.points[1], (255, 0, 255), 2)
        cv2.line(frame, Calibration.points[0], Calibration.points[2], (255, 0, 255), 2)
        cv2.line(frame, Calibration.points[2], Calibration.points[3], (255, 0, 255), 2)
        cv2.line(frame, Calibration.points[3], Calibration.points[1], (255, 0, 255), 2)

        # results = model.predict(frame, classes=[0, 15])  # predict on an image3
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.tolist()
        else:
            continue

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id, cls in zip(boxes, track_ids, classes):
            if cls != 0.0: continue

            x, y, w, h = box
            y += h / 2

            track = track_history[track_id]
            track_warped = track_history_warped[track_id]

            if len(track) > 2:  # Calculate track velocity
                x2, y2 = track[-1][0], track[-1][1]
                dist = math.sqrt(math.pow(x - x2, 2) + math.pow(y - y2, 2))
                vel = dist * fps
                orientation = math.atan((y2 - y) / (x2 - x))
                cv2.arrowedLine(annotated_frame,
                                (int(x), int(y)),
                                (int(x2 - (x2 - x) * 10), int(y2 - (y2 - y) * 10)),
                                (0, 0, 255), 3)
                cv2.putText(annotated_frame,
                            " Vel" + str(int(vel)) + "pix/s",
                            (int(x + w / 2), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 230, 230))

            # Bird coordinates transformation
            bird_vec = np.matmul(perspective_matrix, np.array([x, y, 1]))
            bird_coords = (int(bird_vec[0] / bird_vec[2]), int(bird_vec[1] / bird_vec[2]))
            # print('Coords:', [int(x), int(y)], '->', bird_coords)
            # bird_map = cv2.circle(bird_map, bird_coords, 4, (0, 255, 0), 2)

            track.append((float(x), float(y)))  # x, y center-ground point
            track_warped.append(bird_coords)

            if len(track) > 30:
                track.pop(0)
                track_warped.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=3)

            points_warped = np.hstack(track_warped).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(bird_map, [points_warped], isClosed=False, color=(230, 230, 230), thickness=3)

            cv2.putText()

        writer.write(annotated_frame)

        # Uncomment to see the unwrapped image
        if SHOW:
            # frame_warped = cv2.warpPerspective(frame, perspective_matrix, (600,800))
            # cv2.imshow('Video', frame_warped) # Uncomment to see bird perspective image
            cv2.imshow('Map', bird_map)
            cv2.imshow('Camera view', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    else:
        break
cap.release()
writer.release()
cv2.destroyAllWindows()
