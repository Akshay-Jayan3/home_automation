import cv2
import mediapipe as mp
import time



cmp = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# results = hands.process(imgRGB)

def findHands(img, draw=True):

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img


# def findPossition(img,handNo=0 ,draw=True):
#     lmList = []
#     if results.multi_hand_landmarks:
#         for handLms in results.mul
#         myHand = results.multi_hand_landmarks[handNo]

#         for id, lm in enumerate(myHand.landmark):
#             h, w, c = img.shape
#             cx, cy = int(lm.x*w), int(lm.y*h)
#             lmList.append([id, cx, cy])
            
#             if draw:
#                 cv2.circle(img, (cx, cy), 7, (255, 0, 0), 
#                 cv2.FILLED)
#     return lmList

pTime = 0
cTime = 0
tipIds = [4, 8, 12, 16, 20]
while True:
    success, img = cmp.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    img = findHands(img)
    #lmList = findPossition(img)
    if results.multi_hand_landmarks:
        for hand_index, hand_info in enumerate(results.multi_handedness):
            hand_label = hand_info.classification[0].label
            myHand = results.multi_hand_landmarks[hand_index]
            lmList = []
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
            print(hand_label, lmList)



    # if len(lmList) != 0:
    #     fingers = []
    #     # Thumb
    #     if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
    #         fingers.append(1)
    #     else:
    #         fingers.append(0)
    #     # fingers except thumb
    #     for id in range(1, 5):
    #         if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
    #             fingers.append(1)
    #         else:
    #             fingers.append(0)
    #     print(fingers)



    # if len(lmList) != 0:
    #     print(lmList)
    
    cTime = time.time()
    fps = 1/ (cTime-pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 8, 255), 3)

    cv2.imshow('Image', img)
    
    if cv2.waitKey(1) == 27:
        break

cmp.release()
cv2.destroyAllWindows()

