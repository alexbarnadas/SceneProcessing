import numpy as np
import cv2
import time


def get_cameras_list():
    cam_list = [
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

    return cam_list
def get_sources():
    cameras = get_cameras_list()
    rtsp_username = "ODIN"
    rtsp_password = "**********"
    rtsp_stream_type = "m"  # Use "u" for unicast, "m" for multicast
    rtsp_stream_path = "defaultSecondary"
    # Use "defaultPrimary" for primary stream (provided by Avigilon Control Center),
    # "defaultSecondary" for secondary stream, or "defaultTertiary" for tertiary stream
    # Build the RTSP URL with the parameters

    sources = []
    for i in range(len(cameras)):
        rtsp_ip = cameras[i]
        sources.append(f"rtsp://{rtsp_username}:{rtsp_password}@{rtsp_ip}/rtsp/{rtsp_stream_path}?streamType={rtsp_stream_type}")

def resize_dual_camera_image(source_url):
    cap = cv2.VideoCapture(source_url)
    if not cap.isOpened():
        #return None
        return source_url  # Cannot open source, do nothing

    # Defines the new dimensions
    new_width = 768
    new_height = 432

    frames = []
    frame_count = 0  # Processed frames counter
    out_url = f"output_{int(time.time())}.avi"  # Unique file name based on the timestamp

    while frame_count < 100:  # Limits to process only 100 frames as an example
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (new_width, new_height))
        frames.append(frame)
        frame_count += 1

    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_url, fourcc, 25.0, (new_width, new_height))

    for frame in frames:
        out.write(frame)

    out.release()

    return out_url  # Returns the URL of the resized font

def get_camera_index(camera_name):

    camera_list = get_cameras_list()

    # get the index of the camera

    for i in range(len(camera_list)):
        if camera_list[i] == camera_name:
            return i

    return -1


def get_camera_name(camera_index):
    camera_list = get_cameras_list()

    return camera_list[camera_index]


def convert_image(image, new_shape=(256, 128, 3)):
    # check if image is a numpy array
    if type(image) == np.ndarray:
        image = cv2.resize(image, (new_shape[1], new_shape[0]))
        image = image.tostring()
    else:
        image = np.fromstring(image, dtype=np.uint8)
        image = image.reshape(new_shape)

    return image


# Old unused functions
# ¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨
'''
def getCameraExits(idCamera):
    pass


def getIdUser(firstName, lastName):
    pass


def getUserName(idUser):
    pass


def insertAlert(idCamera, dateTime, type, idUser):
    pass


def getLastAlert(idUser, idCamera, type):
    pass


def insertTrackData(idTrack, idCamera):
    conn, cur = createConnection()

    cur.execute("INSERT INTO tracks (idTrack, idCamera) VALUES (?, ?)", (idTrack, idCamera))
    conn.commit()
    conn.close()


def getTrackData(idTrack):
    conn, cur = createConnection()
    idUser = None
    cur.execute("SELECT idUser, idCamera FROM tracks WHERE idTrack = ?", (idTrack,))
    idUser, idCamera = cur.fetchone()[:]

    name = ""
    if idUser != None:
        cur.execute("SELECT firstName, lastName FROM users WHERE idUser = ?", (idUser,))
        firstName, lastName = cur.fetchone()
        name = firstName + " " + lastName

    # get all the bboxes and names of the track
    bboxes = []
    cur.execute("SELECT x1, y1, x2, y2 FROM reid WHERE idTrack = ?", (idTrack,))
    for x1, y1, x2, y2 in cur:
        bboxes.append(np.array([x1, y1, x2, y2]))

    conn.close()
    return name, idCamera, bboxes

def getReidData(activeTracks):
    if len(activeTracks) == 0:
        return []

    conn, cur = createConnection()

    query = "SELECT image, idTrack FROM reid WHERE idTrack NOT IN (%s)" % ','.join(map(str, activeTracks))

    cur.execute(query)
    reids = []
    for image, idTrack in cur:
        image = convertImage(image)
        reids.append((idTrack, image))
    conn.close()
    return reids


# delete images from database for the id self.id
def deleteReidData(idTrack):
    conn, cur = createConnection()
    cur.execute("DELETE FROM reid WHERE idTrack = ?", (idTrack,))
    conn.commit()
    conn.close()


def insertReidData(idTrack, image, bbox):
    conn, cur = createConnection()
    image = convertImage(image)
    x1, y1, x2, y2 = bbox

    cur.execute("INSERT INTO reid (image, idTrack, x1, y1, x2, y2) VALUES (?, ?, ?, ?, ?, ?)",
                (image, idTrack, int(x1), int(y1), int(x2), int(y2)))
    conn.commit()
    conn.close()


def truncateTable(table):
    conn, cur = createConnection()

    cur.execute("TRUNCATE TABLE " + table)

    conn.commit()
    conn.close()


def truncateTracksData():
    conn, cur = createConnection()
    cur.execute("SELECT idTrack FROM tracks")
    idTracks = cur.fetchall()

    for idTrack in idTracks:
        cur.execute("DELETE FROM tracks WHERE idTrack = ?", (idTrack[0],))

    conn.commit()
    conn.close()


def truncateReidData():
    truncateTable("reid")


def getAdminNumbers():
    conn, cur = createConnection()

    cur.execute("SELECT phone_number, prefix FROM admins")

    numbers = cur.fetchall()

    conn.close()

    return numbers
'''
