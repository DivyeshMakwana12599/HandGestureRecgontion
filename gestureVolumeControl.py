import HandTrackingModule as htm
import cv2 as cv
import mediapipe as mp
import time


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
        p = mapVal(distance, scale/7, scale, 0, 100)
    return p


def mapVal(x, in_min, in_max, out_min, out_max):
    return ((x - in_min) * (out_max - out_min)) / ((in_max - in_min) + out_min)


def drawSlider(x, img):
    cv.rectangle(img)
    pass


def main():
    vol = 0
    pTime = cTime = 0
    Running = True
    cap = cv.VideoCapture(0)
    dectector = htm.handDetector()
    while Running:
        success, img = cap.read()
        img = dectector.findHands(img)
        print(img.shape)
        finger_xy = dectector.findPosition(img)
        if finger_xy:
            vol = gestureSliderNum(finger_xy)
        # drawSlider(vol, img)
        cv.rectangle(img, (600, 340), (620, 440), (177, 132, 91), 2)
        cv.rectangle(img, (600, 440 - int(vol)),
                     (620, 440), (106, 118, 252), -1)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, 'FPS: ' + str(int(fps)), (5, 15), cv.FONT_HERSHEY_PLAIN, 1,
                   (0, 0, 0), 2)
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            Running = False
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
