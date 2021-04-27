import cv2
import mediapipe as mp
import time

running = True
cap = cv2.VideoCapture(0)

myHands = mp.solutions.hands
hands = myHands.Hands()




while running:
    sucess , img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results =  hands.process(imgRGB)
    print(results)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        cv2.destroyAllWindows()