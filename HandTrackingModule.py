import cv2 as cv
import mediapipe as mp
import time


volumeWIN = [
    -65.25,
    -56.992191314697266,
    -51.671180725097656,
    -47.73759078979492,
    -44.61552429199219,
    -42.0267333984375,
    -39.81534194946289,
    -37.88519287109375,
    -36.17274856567383,
    -34.63383865356445,
    -33.236507415771484,
    -31.956886291503906,
    -30.776674270629883,
    -29.681533813476562,
    -28.660018920898438,
    -27.702852249145508,
    -26.802404403686523,
    -25.952329635620117,
    -25.14728546142578,
    -24.382741928100586,
    -23.65481948852539,
    -22.96017074584961,
    -22.295883178710938,
    -21.65941619873047,
    -21.04853057861328,
    -20.46125030517578,
    -19.89581871032715,
    -19.35066795349121,
    -18.824392318725586,
    -18.315731048583984,
    -17.82354164123535,
    -17.346792221069336,
    -16.884540557861328,
    -16.43593406677246,
    -16.000186920166016,
    -15.57658576965332,
    -15.164469718933105,
    -14.76323127746582,
    -14.372313499450684,
    -13.991198539733887,
    -13.619404792785645,
    -13.256488800048828,
    -12.902036666870117,
    -12.55566120147705,
    -12.217001914978027,
    -11.885725975036621,
    -11.561514854431152,
    -11.244073867797852,
    -10.933128356933594,
    -10.628416061401367,
    -10.329692840576172,
    -10.036725044250488,
    -9.749299049377441,
    -9.467207908630371,
    -9.190256118774414,
    -8.91826057434082,
    -8.651045799255371,
    -8.388448715209961,
    -8.13031005859375,
    -7.876482963562012,
    -7.626824855804443,
    -7.381200790405273,
    -7.1394829750061035,
    -6.901548862457275,
    -6.6672821044921875,
    -6.436570644378662,
    -6.209307670593262,
    -5.98539400100708,
    -5.764730453491211,
    -5.547224998474121,
    -5.33278751373291,
    -5.121333599090576,
    -4.912779808044434,
    -4.707049369812012,
    -4.5040669441223145,
    -4.3037590980529785,
    -4.1060566902160645,
    -3.9108924865722656,
    -3.718202590942383,
    -3.527923583984375,
    -3.339998245239258,
    -3.1543679237365723,
    -2.970977306365967,
    -2.7897727489471436,
    -2.610703229904175,
    -2.4337174892425537,
    -2.2587697505950928,
    -2.08581280708313,
    -1.9148017168045044,
    -1.7456932067871094,
    -1.5784454345703125,
    -1.4130167961120605,
    -1.2493702173233032,
    -1.0874667167663574,
    -0.9272695183753967,
    -0.768743097782135,
    -0.6118528842926025,
    -0.4565645754337311,
    -0.30284759402275085,
    -0.15066957473754883,
    0.0,
]


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=False):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            for lm in myHand.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([cx, cy])
                if draw:
                    cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)
        return lmList


def main():
    pTime = 0
    cTime = 0
    Running = True
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while Running:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3,
                   (255, 0, 255), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            Running = False
            cv.destroyAllWindows()


if __name__ == "__main__":
    main()
