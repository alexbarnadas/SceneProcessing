import cv2
import numpy as np
import yaml


class SceneCalibration:
    def __init__(self, img, mat=None, proportion=1):
        self.img = img

        self.points = []  # Must be clicked
        self.real_w2d_proportion = proportion
        self.perspective_matrix = mat
        self.bird_view = None

        self.get_input()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_input(self):
        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('image', self.img)

        cv2.setMouseCallback('image', self.click_event)

        '''
        print('Give the dimensions of the scene with the real distance between the points specified:')
        real_width = float(input('Width is how far are in the scene the 1st and 2nd points: '))
        real_depth = float(input('Depth is how far are in the scene the 1st and 3rd points: '))
        self.real_w2d_proportion = real_width/real_depth
        '''

    def click_event(self, event, x, y, flags, params):

        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:

            print(x, ' ', y)
            self.points.append([x, y])
            if len(self.points) == 4:
                self.get_perspective()

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' + str(y), (x, y), font, 1, (255, 0, 0), 2)
            cv2.circle(self.img, (x, y), 3, (0, 0, 255), -1)
            cv2.imshow('image', self.img)

    def get_perspective(self):
        width = self.points[1][0] - self.points[0][0]
        depth = self.points[2][1] - self.points[0][1]
        depth = width * self.real_w2d_proportion
        pts1 = np.float32(self.points[0:4])
        pts2 = np.float32([[0, 0], [width, 0], [0, depth], [width, depth]])

        self.perspective_matrix = cv2.getPerspectiveTransform(pts1, pts2)
        bird_view = cv2.warpPerspective(self.img, self.perspective_matrix, (width, depth))

        cv2.imshow('output', bird_view)

    def scene2real(self, x, y):
        transposed_coordinates = np.dot(self.perspective_matrix, np.array([x, y, 1]).T)
        return transposed_coordinates  # (x,y,scale)

    def store_calibration(self, name):
        matrix = yaml.safe_load(self.perspective_matrix)

        with open('perspective_matrixes.yaml', 'w') as file:
            yaml.dump(matrix, file)

        print(open('names.yaml').read())


def calibrate(source, scene_proportion=1):

    f = cv2.VideoCapture(source)  # './Terraza.MOV')

    success, frame = f.read()
    if not success:
        print('ERROR: Video could not be loaded while attempting the calibration.')
        exit()

    calibration = SceneCalibration(frame, proportion=scene_proportion)
    print(calibration.perspective_matrix)

    # Load video and detect, then transform the data with
    x = 1082
    y = 746
    new_coords = calibration.scene2real(x, y)

    print('\n')
    print(new_coords)

    return calibration


if __name__ == "__main__":
    calibrate('./TestVideos/Inetum_cam1.mov')
