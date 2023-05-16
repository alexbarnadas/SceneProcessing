import cv2
import numpy as np

class SceneCalibration():
    def __init__(self, img):
        self.img = img

        self.points = [] # Must be clicked 
        self.real_w2d_proportion = 1
        self.perspective_matrix = None 
        self.bird_view = None

        self.getInput()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def getInput(self):
        cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
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
            self.points.append([x,y])
            if len(self.points)==4: 
                self.get_perspective()

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +str(y), (x,y), font,1, (255, 0, 0), 2)
            cv2.circle(self.img, (x,y),4,(0,0,255),-1)
            cv2.imshow('image', self.img)

    def get_perspective(self):

        width = self.points[1][0] - self.points[0][0]
        depth = self.points[2][1] - self.points[0][1]
        depth  = width * self.real_w2d_proportion
        pts1 = np.float32(self.points[0:4])
        pts2 = np.float32([[0,0], [width,0], [0,depth], [width, depth]])

        self.perspective_matrix = cv2. getPerspectiveTransform(pts1, pts2)
        bird_view = cv2.warpPerspective(self.img, self.perspective_matrix, (width, depth))

        cv2.imshow('output', bird_view)
    
    def scene2real(self, x,y):
        transposed_coordinates = np.dot(self.perspective_matrix, np.array([x,y,1]).T)
        return transposed_coordinates # (x,y,scale)

if __name__=="__main__":
    f = cv2.VideoCapture('./Terraza.MOV')
    rval, frame = f.read()
    img = frame #cv2.imread('.\standard_test_images\peppers.png', 1)

    Calibration = SceneCalibration(img)
    print(Calibration.perspective_matrix)

    # Load video and detect, then transform the data with
    x = 1082
    y = 746
    new_coords = Calibration.scene2real(x,y)

    print('\n')
    print(new_coords)
    