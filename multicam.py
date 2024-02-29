from collections import defaultdict
import threading
from threading import Semaphore
from datetime import datetime
import numpy as np
import math
import yaml
import csv
import cv2
import pandas as pd
from ultralytics import YOLO
from calibrator import *
from camera_functions import *

import config

# from alerts import KafkaMessager

# kfk = KafkaMessager()
dt = 20  # number of frames to calculate mean velocity
history_save_interval = (10 * 30)  # Save every 10 seconds at 30 fps (300 frames)


def run_tracker_in_thread(source, stream_id, perspective_matrix=None, save=False, show=False):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        source (str): The path to the video file or the identifier for the webcam/external camera source.
        stream_id (int): An index to uniquely identify the file being processed, used for display purposes.


    Note:
        Press 'q' to quit the video display window.

    Next Steps:
        Allow show multiple windows:
        https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/
    """
    semaforo = Semaphore(1)
    camera_list = get_cameras_list()
    # Examples, delete in the future ############
    # camera_list = [0]
    sources_list = ['TestVideos/Inetum_cam1.mov',
                    'TestVideos/Inetum_cam2.mov']
    camera_list = sources_list

    track_history = defaultdict(lambda: [])
    track_history_warped = defaultdict(lambda: [])
    pax_history = []
    interval = 90
    x_pad, y_pad = 100, 100
    x_map_size, y_map_size = 600, 600
    line_color = (250, 0, 250)

    model = YOLO('Models/yolov8n-pose.pt')

    cap = cv2.VideoCapture(source)
    frame_size, fps = (int(cap.get(3)), int(cap.get(4))), cap.get(5)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter('Demo.mp4', fourcc, fps, frame_size)
    writer_map = cv2.VideoWriter('Demo_map.mp4', fourcc, fps, (x_map_size+2*x_pad, y_map_size+2*y_pad))

    # Scene and camera calibration
    success, first_frame = cap.read()
    if not success:
        print('Error while loading calibration frame')
        exit()

    if perspective_matrix is None:
        perspective_matrix = np.array([[1.5911, 4.5363, -1667.1],
                                       [-1.5115, 7.449, -15.472],
                                       [1.6072e05, 0.0044881, 1]])

    bird_map = 255*np.ones((x_map_size + 2*int(x_pad), y_map_size + 2*int(y_pad), 3), dtype=np.uint8)
    map_boundaries = np.array([[x_pad, y_pad],
                               [bird_map.shape[0] - x_pad, y_pad],
                               [bird_map.shape[0] - x_pad, bird_map.shape[1] - y_pad],
                               [x_pad, bird_map.shape[1] - y_pad]])
    cv2.polylines(bird_map,
                  np.int32([map_boundaries]),
                  isClosed=True, color=line_color, thickness=3)

    cv2.imwrite('www.png', bird_map)

    # Output result for Angelo
    people_flow_history = []


    header = ['FrameNumber', 'ScreenCoords', 'SceneCoords', 'Velocity', 'Orientation']
    csvfile = open('PeopleFlowHistory_' + str(stream_id) + '.csv', 'w')
    c = csv.DictWriter(csvfile, header)
    c.writeheader()
    csvfile.close()

    frame_id = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        # Process every 3 frames to promote synchronization
        if frame_id % 3 != 0:
            frame_id += 1
            continue

        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_LINEAR)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        result_generator = model.track(frame,
                                       persist=True,
                                       verbose=False,
                                       stream=True,
                                       tracker="bytetrack.yaml",
                                       half=True)
        #  imgsz=1920, save_txt=True, save_conf=True, save=True, conf=0.25, iou=0.5)   # Other possible parameters

        # Get the boxes and track IDs
        for result in result_generator:
            # Visualize the results on the frame
            annotated_frame = result.plot()
            pax_framed = len(result)

            pax_history.append(pax_framed)
            if len(pax_history) > 90:
                pax_history.pop(0)
                avg_pax_history = sum(pax_history) / interval
                semaforo.acquire()
                config.mean_framed_pax[source] = avg_pax_history
                semaforo.release()

            if result.boxes.id is not None:
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.tolist()
                keypoints = result.keypoints.xy
            else:
                continue

            # Plot the tracks
            bird_map_written = bird_map.copy()
            for box, track_id, cls, keypt in zip(boxes, track_ids, classes, keypoints):
                if cls != 0.0: continue

                x_box, y_box, w, h = box.cpu()
                x_box, y_box, w, h = int(x_box), int(y_box), int(w / 2), int(h / 2)

                # box_img = frame[y_box - h:y_box + h, x_box - w:x_box + w]

                if keypt[15][0] != 0 and keypt[16][0] != 0:
                    feet = (keypt[15] + keypt[16]).div(2).cpu()
                    x, y = feet
                else:
                    continue

                # Bird coordinates transformation
                wx, wy, w_scale = np.matmul(perspective_matrix, np.array([x, y, 1]))
                bird_coords = [int(wx / w_scale) + x_pad, int(wy / w_scale) + y_pad]
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
                else:
                    velocity = orientation = None

                track.append((float(x), float(y)))  # x, y center-ground point
                track_warped.append(bird_coords)

                if len(track) > 30:
                    track.pop(0)
                    track_warped.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=line_color, thickness=3)

                points_warped = np.hstack(track_warped).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(bird_map_written, [points_warped], isClosed=False, color=line_color, thickness=3)

                if save:
                    writer.write(annotated_frame)
                    writer_map.write(bird_map_written)

                people_flow_history.append({
                    'FrameNumber': frame_id,  # id of the current frame
                    'ScreenCoords': box.int().tolist(),  # x,y,h,w
                    'SceneCoords': bird_coords,  # real_x1,real_x2
                    'Velocity': velocity,  # pix/s
                    'Orientation': orientation  # radians
                })

        frame_id += 1
        if frame_id > 1e6: frame_id = 0
        print('CAM' + str(stream_id) + ',' + str(frame_id) + ': ' + str(pax_framed) + 'pax')

        #        if len(people_flow_history) % history_save_interval == 0:
        #            people_flow_df = pd.DataFrame(people_flow_history)
        #            people_flow_df.to_csv('PeopleFlowHistory_' + str(stream_id) + '.csv', mode='a', header=False, index=False)
        #            people_flow_history = []
        if stream_id == 0:
            try:
                total_people = 0
                for camera in camera_list:
                    total_people += config.mean_framed_pax[source]
                print('Total number of people detected: ' + str(total_people))
            except Exception as e:
                print(f'Could not count people in the hospital because of: {e}')

        # Uncomment to see the unwrapped image
        if show:
            frame_warped = cv2.warpPerspective(frame, perspective_matrix, (x_map_size,800))
            cv2.imshow('Video', frame_warped) # Uncomment to see bird perspective image
            cv2.imshow('Camera view', annotated_frame)
            cv2.imshow('Map', bird_map_written)  # Uncomment so see the map

            if cv2.waitKey(1) & 0xFF == ord('q'): break
            if stream_id == 0:
                #            try:
                total_people = 0
                while True:
                    for camera in camera_list:
                        total_people += config.mean_framed_pax[source]
                print('a')
                print(total_people)
    #            except Exception as e:
    #                print(f'Could not count people in the hospital because of: {e}')

    # Release video sources
    cap.release()


if __name__ == "__main__":
    # Load the models
    model1 = YOLO('Models/yolov8n-pose.pt')
    model2 = YOLO('Models/yolov8n-pose.pt')

    # Define the video sources for the trackers
    video_file1 = "TestVideos/Inetum_cam2.mov"  # Path to video file, 0 for webcam
    video_file2 = "TestVideos/Inetum_cam1.mov"  # Path to video file, 0 for webcam, 1 for external camera

    calibration = calibrate(video_file1, 1)
    perspective_matrix = calibration.perspective_matrix

    # Create the tracker threads
    tracker_thread1 = threading.Thread(target=run_tracker_in_thread,
                                       args=(video_file1, 1, perspective_matrix, True, False),
                                       daemon=True, )
    # tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model2, 2), daemon=True)

    # Start the tracker threads
    tracker_thread1.start()
    # tracker_thread2.start()

    # Wait for the tracker threads to finish
    tracker_thread1.join()
    # tracker_thread2.join()

    # Clean up and close windows
    cv2.destroyAllWindows()
