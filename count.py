import cv2
import mediapipe as mp
import time
from collections import Counter
import threading

def findHands(img, results, mpDraw, mpHands, draw=True):

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if draw:
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    return img


# record video for 2 sec and extract finger count from it 
# writing video for 2 sec and read it and obtain count 

# and repeat it again and again

# render page after each condition check finger


def countFinger():
    pTime = time.time()
    t = threading.Timer(10, countFinger)
    t.start()

    cmp = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    mpDraw = mp.solutions.drawing_utils

    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # results = hands.process(imgRGB)


    tipIds = [4, 8, 12, 16, 20]
    while True:

        cTime = time.time()

        if cTime - pTime > 9:
            break

        success, img = cmp.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        img = findHands(img, results, mpDraw, mpHands)
        up_finger_count = 0
        #lmList = findPossition(img)
        if results.multi_hand_landmarks:
            fingers_1 = []
            fingers_2 = []
            for hand_index, hand_info in enumerate(results.multi_handedness):
                hand_label = hand_info.classification[0].label
                myHand = results.multi_hand_landmarks[hand_index]
                lmList = []
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])
                # print(hand_label, lmList)
            
                if hand_label == 'Right' and len(lmList) != 0:

                    
                    if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
                        fingers_1.append(1)
                        
                    else:
                        fingers_1.append(0)
                        
                    
                    for id in range(1, 5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                            fingers_1.append(1)
                            
                        else:
                            fingers_1.append(0)   
                    
                if hand_label == 'Left' and len(lmList) != 0:

                    
                    if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
                        fingers_2.append(1)
                        
                    else:
                        fingers_2.append(0)
                        
                    
                    for id in range(1, 5):
                        if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                            fingers_2.append(1)
                            
                        else:
                            fingers_2.append(0)
                    # print('2:',fingers_2)
                
            # print('1: ', fingers_1)
            # print('2: ', fingers_2)
            finger_list = fingers_1 + fingers_2
            up_finger_count = finger_list.count(1)

        if up_finger_count:
            if up_finger_count > 5:
                t.cancel()
                #break
            print(up_finger_count)
            break

        # print(up_finger_count)

        # if up_finger_count == 2:
        #     break

        cv2.putText(img, str(up_finger_count), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (255, 8, 255), 3)

        cv2.imshow('Image', img)
        
        if cv2.waitKey(1) == 27:
            t.cancel()
            break

    cmp.release()
    cv2.destroyAllWindows()




countFinger()
