import threading
from multicam import *
from camera_functions import *
from calibrator import *

def main(calibration_needed=False):
    # Define the video sources for the trackers

    camera_list = get_cameras_list()
    sources_list = get_sources()

    # Examples, delete in the future ############
    #camera_list = [0]
    sources_list = ['TestVideos/Inetum_cam1.mov',
                   'TestVideos/Inetum_cam2.mov']
    camera_list = sources_list
    #############################################

    tracker_threads = []

    interval = 20  # frames
    mean_framed_pax = {}

    def count_people_thread(camera_list):
        global mean_framed_pax
        total_people = 0
        while True:
            for camera in camera_list:
                total_people += mean_framed_pax[camera]
        print('a')
        print(total_people)


    if calibration_needed:
        perspective_matrixes = []
        for camera_ip in camera_list:
            calibration = calibrate(camera_ip, 1)
            perspective_matrixes.append(calibration.perspective_matrix)

    for camera_id in range(len(camera_list)):
        camera = camera_list[camera_id]
        source = sources_list[camera_id]
        print('Loaded stream ' + str(camera))

        # Create the tracker threads
        tracker_thread = threading.Thread(target=run_tracker_in_thread,
                                          args=(source, camera_id, 20, False, False),
                                          daemon=False, )
        # Start the tracker threads, and stack it
        tracker_thread.start()
        tracker_threads.append(tracker_thread)

    people_counter_thread = threading.Thread(target=count_people_thread,
                                             args=(camera_list),
                                             daemon=False)
    people_counter_thread.start()

    join = True
    if join:
        # Wait for the tracker threads to finish
        for tracker_thread in tracker_threads:
            tracker_thread.join()
        people_counter_thread.join()

    # Clean up and close windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
