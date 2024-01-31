import threading
from ultralytics import YOLO
from multicam import *
from camera_functions import *


def main():
    # Define the video sources for the trackers
    camera_list = [
        "192.168.1.11",
        "192.168.1.12",
        "192.168.1.13",
        "192.168.1.14",
        "192.168.1.15",
        "192.168.1.16",
        "192.168.1.17",
        "192.168.1.18",
        "192.168.1.19",
        "192.168.1.20"
    ]

    camera_list = [f"0"]
    moddels = []

    for camera_IP in camera_list:
        print(camera_IP)
        # Load the models
        models[camera_IP] = YOLO('Models/yolov8n-pose.pt')

        # Create the tracker threads
        tracker_thread = threading.Thread(target=run_tracker_in_thread,
                                          args=(video_file1, models[camera_IP], 1, False, True),
                                          daemon=True, )
        tracker_threads.append(tracker_thread)

        # Start the tracker threads
        tracker_thread.start()

        # Wait for the tracker threads to finish
        for tracker_thread in tracker_threads:
            tracker_thread.join()

    # Clean up and close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
