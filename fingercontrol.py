import cv2
import mediapipe as mp
import time
from collections import Counter
from urllib.request import urlopen
import json
import requests


username = 'sagarrb123'
url_get = 'http://ghome123.herokuapp.com/api/{}/get?format=json'.format(username)
url = "https://ghome123.herokuapp.com/api/sagarrb123/put"

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


def changeVal(up_finger_count, username):
    response = urlopen(url_get)
    data_json = json.loads(response.read())
    sw_1 = data_json['switch_1']
    sw_2 = data_json['switch_2']
    sw_3 = data_json['switch_3']
    sw_4 = data_json['switch_4']
    sw_5 = data_json['switch_5']
    sw_6 = data_json['switch_6']
    if up_finger_count == 1:
        sw_1 = not sw_1
    elif up_finger_count == 2:
        sw_2 = not sw_2
    elif up_finger_count ==3:
        sw_3 = not sw_3
    elif up_finger_count == 4:
        sw_4 = not sw_4 
    elif up_finger_count == 5:
        sw_5 = not sw_5
    elif up_finger_count == 6:
        sw_6 = not sw_6
    else:
        pass
    return {"username": username,"switch_1": sw_1,"switch_2": sw_2,"switch_3": sw_3,"switch_4": sw_4,"switch_5": sw_5, "switch_6": sw_6}    




tipIds = [4, 8, 12, 16, 20]
prev_count = 0
while True:
    success, img = cmp.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    img = findHands(img)
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


        
        if up_finger_count != prev_count:
            prev_count = up_finger_count
            #print(up_finger_count)

            postData = changeVal(up_finger_count, username)
            x = requests.post(url, json=postData)
            print(x.text)

            




    cv2.putText(img, str(up_finger_count), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 8, 255), 3)

    cv2.imshow('Image', img)
    
    if cv2.waitKey(1) == 27:
        break

cmp.release()
cv2.destroyAllWindows()

