import HandTrackingModule as htm
import cv2 as cv
import mediapipe as mp
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math



def findLen(x, y):
    return (((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2)) ** 0.5


def gestureSliderNum(lmList):
    scale = findLen(lmList[1], lmList[4])
    distance = findLen(lmList[4], lmList[8])
    if scale < distance:
        p = 100
    elif (scale/6) > distance:
        p = 0
    else:
        p = mapVal(distance, scale/6, scale, 0, 100)
    return p


def mapVal(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


def drawSlider(x, img):
    cv.rectangle(img)
    pass


def main():
    vol = volDB = 0
    pTime = cTime = 0
    timeDuration = time.time()
    Running = True
    cap = cv.VideoCapture(0)
    dectector = htm.handDetector(detectionCon=0.3)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    winVolume = volume.GetMasterVolumeLevel()
    while Running:
        success, img = cap.read()
        img = dectector.findHands(img)
        finger_xy = dectector.findPosition(img)
        if finger_xy:
            if (time.time() - timeDuration) > 0.5:
                volDB = htm.volumeWIN[int(vol)]
                vol = gestureSliderNum(finger_xy)
                timeDuration = time.time()        
        # drawSlider(vol, img)
        cv.rectangle(img, (600, 340), (620, 440), (177, 132, 91), 2)
        cv.rectangle(img, (600, 440 - int(vol)),
                     (620, 440), (106, 118, 252), -1)
        cv.putText(img, 'Vol: ' + str(int(vol)) + '%', (555, 460), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        volume.SetMasterVolumeLevel(volDB, None)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, 'FPS: ' + str(int(fps)), (5, 15), cv.FONT_HERSHEY_PLAIN, 1,
                   (0, 0, 0), 2)
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            Running = False
            cv.destroyAllWindows()
            volume.SetMasterVolumeLevel(winVolume, None)


if __name__ == "__main__":
    main()
