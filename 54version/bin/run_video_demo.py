import cv2
import argparse
import glob
import numpy as np
import os
import time
import sys
sys.path.append(os.getcwd())
from siamfc import SiamRPNTracker

# Drawing constants
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 480
PADDING = 2

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

drawnBox = np.zeros(4)
boxToDraw = np.zeros(4)
mousedown = False
mouseupdown = False
initialize = False
def on_mouse(event, x, y, flags, params):
    global mousedown, mouseupdown, drawnBox, boxToDraw, initialize
    if event == cv2.EVENT_LBUTTONDOWN:
        drawnBox[[0,2]] = x
        drawnBox[[1,3]] = y
        mousedown = True
        mouseupdown = False
    elif mousedown and event == cv2.EVENT_MOUSEMOVE:
        drawnBox[2] = x
        drawnBox[3] = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawnBox[2] = x
        drawnBox[3] = y
        mousedown = False
        mouseupdown = True
        initialize = True
    boxToDraw = drawnBox.copy()
    boxToDraw[[0,2]] = np.sort(boxToDraw[[0,2]])
    boxToDraw[[1,3]] = np.sort(boxToDraw[[1,3]])


def show_webcam(video_path,mirror=False):
    global tracker, initialize, mouseupdown, boxToDraw
    #cam = cv2.VideoCapture('video.mp4')
    cam = cv2.VideoCapture(video_path)
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam', OUTPUT_WIDTH, OUTPUT_HEIGHT)
    cv2.setMouseCallback('Webcam', on_mouse, 0)
    frameNum = 0
    outputDir = None
    if RECORD:
        print('saving')
        if not os.path.exists('outputs'):
            os.mkdir('outputs')
        tt = time.localtime()
        outputDir = ('outputs/%02d_%02d_%02d_%02d_%02d/' %
                (tt.tm_mon, tt.tm_mday, tt.tm_hour, tt.tm_min, tt.tm_sec))
        os.mkdir(outputDir)
        labels = open(outputDir + 'labels.txt', 'w')
    paused = False
    while True:
        ret_val, img = cam.read()
        if img is None:
            # End of video.
            break
        if mirror:
            img = cv2.flip(img, 1)
        origImg = img.copy()
        drawImg = img.copy()
        while frameNum == 0 or mousedown or paused:
            drawImg = img.copy()
            if mousedown:
                if RECORD:
                    cv2.circle(drawImg, (int(drawnBox[2]), int(drawnBox[3])), 10, [255,0,0], 4)
            elif mouseupdown:
                paused = False
                frameNum += 1
            cv2.rectangle(drawImg,
                    (int(boxToDraw[0]), int(boxToDraw[1])),
                    (int(boxToDraw[2]), int(boxToDraw[3])),
                    [0,0,255], PADDING)
            cv2.imshow('Webcam', drawImg)
            cv2.waitKey(1)

        if initialize:
            # boxToDraw(xmin,ymin,xmax,ymax)
            # bbox(xmin,ymin,width,height)
            bbox = [boxToDraw[0], #xmin
                    boxToDraw[1], #ymin
                    boxToDraw[2]-boxToDraw[0], #width
                    boxToDraw[3]-boxToDraw[1]] #height
            tracker.init(img[:,:,::-1], bbox)
            initialize = False
        else:
            bbox,score = tracker.update(img[:,:,::-1])
            # boxToDraw(xmin,ymin,xmax,ymax)
            # bbox(center_x,center_y,width,height)
            boxToDraw = [int(bbox[0]-bbox[2]/2), 
                         int(bbox[1]-bbox[3]/2),
                         int(bbox[0]+bbox[2]/2), 
                         int(bbox[1]+bbox[3]/2)]
        cv2.rectangle(drawImg,
                (int(boxToDraw[0]), int(boxToDraw[1])),
                (int(boxToDraw[2]), int(boxToDraw[3])),
                [0,0,255], PADDING)
        cv2.imshow('Webcam', drawImg)
        if RECORD:
            if boxToDraw is not None:
                labels.write('%d %.2f %.2f %.2f %.2f\n' %
                        (frameNum, boxToDraw[0], boxToDraw[1],
                            boxToDraw[2], boxToDraw[3]))
            cv2.imwrite('%s%08d.jpg' % (outputDir, frameNum), origImg)
            print('saving')
        keyPressed = cv2.waitKey(1)
        if keyPressed == 27 or keyPressed == 1048603:
            break  # esc to quit
        elif keyPressed != -1:
            paused = True
            mouseupdown = False
        frameNum += 1
    cv2.destroyAllWindows()



# Main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Show the Webcam demo.')
    parser.add_argument('-r', '--record', action='store_true', default=False)
    parser.add_argument( '--video_path', dest='video_path',type=str, default=None)
    parser.add_argument( '--model_path', dest='model_path',type=str, default=None)
    args = parser.parse_args()
    RECORD = args.record
    #tracker = re3_tracker.Re3Tracker()
    tracker = SiamRPNTracker(args.model_path)
    show_webcam(args.video_path, mirror=False)
