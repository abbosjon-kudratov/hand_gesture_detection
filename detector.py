import cv2
import numpy as np
import threading
import imutils
import matplotlib.pyplot as plt
import math
import os
import glob
import re



kernel = np.ones((3, 3), np.uint8)

#read multiple images from directory
images = [cv2.imread(file) for file in glob.glob("./*.png")]

#image names
choose = images[0]
americano_buy = images[1]
latte_buy = images[2]
sugar = images[3]
end = images[4]
americano = images[5]
latte = images[6]
cappucino = images[7]
cappucino_buy = images[8]
welcome = images[9]


'''
to help around some legend goes here:

coffee: prices
1- cappucino = 3$
2- americano = 4$
3- latte = 5$

sugar: 
one sugar = 0.25$
two sugar = 0.50$

P.S. all prices, services and products are not real, just for a testing case

hand gestures:
0=ok
1=one
2=two
3=three
4=four
5=hello/palm
6=other


'''

coffe_names =['Cappucino','Americano','Latte']
coffee_price = {'Cappucino': 3,'Americano': 4, 'Latte': 5}
hand_gesture = 0
longevity = 0



print(coffee_price.get(coffe_names[0]))



# some playaround file text file read/write

file1 = open("file1.txt","r+")
longevity = int(float(file1.readline()))
# print (longevity)
print(type(longevity))
longevity += 1

file1 = './file1.txt'
longevity_str = str(float(longevity))
# print(longevity_str)
with open(file1, 'w') as filetowrite:
    filetowrite.write(longevity_str)




longevity=0

def start(gesture_code):
    '''
   first to start we need to detect hand gesture with 5 fingers or hello
   and proceed
   '''
    global longevity
    # length = 0

    print(longevity)
    # gesture_length = +1
    print()
    if(gesture_code == 0):
        print('failed to load!')
        cv2.imshow('welcome', welcome)


    if(gesture_code == 1):
        # cv2.imshow('gesture_code', welcome)
        longevity += 1
        cv2.waitKey(1)

    if (longevity >= 15):
        cv2.imshow('choose your stuff', choose)
        # longevity += 1
        cv2.waitKey(1)


    # elif(gesture_code == 2):
    #     cv2.imshow('gesture_code', sugar)
    #     cv2.waitKey(1)
    # elif(gesture_code == 3):
    #     cv2.imshow('gesture_code', welcome)
    #     cv2.waitKey(latte)
    # elif(gesture_code == 5 or gesture_code == 4):
    #     cv2.imshow('gesture_code', choose)
    #     cv2.waitKey(1)

    else:
        cv2.imshow('gesture_code', welcome)
        cv2.waitKey(1)






def buy_coffee(gesture_code, frame):
    somethung = frame










def contours_convex_hull(mask,roi,frame):
    # find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find contour of max area(hand)
    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    # approx the contour a little
    epsilon = 0.0005 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # make convex hull around hand
    hull = cv2.convexHull(cnt)

    # define area of hull and area of hand
    areahull = cv2.contourArea(hull)
    areacnt = cv2.contourArea(cnt)

    # find the percentage of area not covered by hand in convex hull
    arearatio = ((areahull - areacnt) / areacnt) * 100

    # find the defects in convex hull with respect to hand
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)

    find_defects(defects,approx,roi,areacnt, arearatio, frame)


def find_defects(defects, approx, roi,areacnt, arearatio, frame):
    l=0
    # code for finding no. of defects due to fingers
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt = (100, 180)

        # find length of all sides of triangle
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        s = (a + b + c) / 2
        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

        # distance between point and convex hull
        d = (2 * ar) / a

        # apply cosine rule here
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        if angle <= 90 and d > 30:
            l += 1
            cv2.circle(roi, far, 3, [255, 0, 0], -1)

        # draw lines around hand
        cv2.line(roi, start, end, [0, 0, 255], 2)
    text_for_detection(l+1,areacnt,arearatio,frame)
    # return l+1


def process_image(img):
    if img is not None:
        # define region of interest
        roi = frame[100:500, 100:500]

        cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 0)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # define range of skin color in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # extract skin color image
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # extrapolate the hand to fill dark spots within
        mask = cv2.dilate(mask, kernel, iterations=4)

        # blur the image
        mask = cv2.GaussianBlur(mask, (5, 5), 100)

        return mask


def text_for_detection(l,areacnt, arearatio,frame):
    font = cv2.FONT_HERSHEY_SIMPLEX

    texts = ['Reposition!',
             'Put hand in the box',
             'Like',
             'OKAY']
    hand_gesture = 0
    if l == 1:
        if areacnt < 2000:
            cv2.putText(frame, texts[1], (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            if arearatio < 12:
                cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            elif arearatio < 17.5:
                cv2.putText(frame, texts[2], (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            else:
                cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
                hand_gesture = 1

    elif l == 2:
        cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        hand_gesture = 2

    elif l == 3:

        if arearatio < 27:
            cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            hand_gesture = 3
        else:
            cv2.putText(frame, texts[3], (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            hand_gesture = 6

    elif l == 4:
        cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        hand_gesture = 4
        timer = +1

    elif l == 5:
        cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        hand_gesture = 5
        timer = +1

    elif l == 6:
        cv2.putText(frame, texts[0], (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        hand_gesture = 7

    else:
        cv2.putText(frame, texts[0], (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    start(hand_gesture)






cap = cv2.VideoCapture(0)
while True:

    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
        # therefore this try error statement

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        roi = frame[100:500, 100:500]
        mask = process_image(frame)

        contours_convex_hull(mask,roi,frame)

        # # find contours
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #
        # # find contour of max area(hand)
        # cnt = max(contours, key=lambda x: cv2.contourArea(x))
        #
        # # approx the contour a little
        # epsilon = 0.0005 * cv2.arcLength(cnt, True)
        # approx = cv2.approxPolyDP(cnt, epsilon, True)
        #
        # # make convex hull around hand
        # hull = cv2.convexHull(cnt)
        #
        # # define area of hull and area of hand
        # areahull = cv2.contourArea(hull)
        # areacnt = cv2.contourArea(cnt)
        #
        # # find the percentage of area not covered by hand in convex hull
        # arearatio = ((areahull - areacnt) / areacnt) * 100
        #
        # # find the defects in convex hull with respect to hand
        # hull = cv2.convexHull(approx, returnPoints=False)
        # defects = cv2.convexityDefects(approx, hull)

        # l = find_defects(defects,roi)
        # l = no. of defects
        # l = 0

        # # code for finding no. of defects due to fingers
        # for i in range(defects.shape[0]):
        #     s, e, f, d = defects[i, 0]
        #     start = tuple(approx[s][0])
        #     end = tuple(approx[e][0])
        #     far = tuple(approx[f][0])
        #     pt = (100, 180)
        #
        #     # find length of all sides of triangle
        #     a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        #     b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        #     c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        #     s = (a + b + c) / 2
        #     ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
        #
        #     # distance between point and convex hull
        #     d = (2 * ar) / a
        #
        #     # apply cosine rule here
        #     angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        #
        #     # ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
        #     if angle <= 90 and d > 30:
        #         l += 1
        #         cv2.circle(roi, far, 3, [255, 0, 0], -1)
        #
        #     # draw lines around hand
        #     cv2.line(roi, start, end, [0, 0, 255], 2)
        #
        # l += 1

        # print corresponding gestures which are in their ranges

        # text_for_detection(l, areacnt, arearatio,frame)

        # font = cv2.FONT_HERSHEY_SIMPLEX
        # if l == 1:
        #     if areacnt < 2000:
        #         cv2.putText(frame, 'Put hand in the box', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #     else:
        #         if arearatio < 12:
        #             cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #         elif arearatio < 17.5:
        #             cv2.putText(frame, 'Best of luck', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #
        #         else:
        #             cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #
        # elif l == 2:
        #     cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #     hand_gesture = 2
        #
        # elif l == 3:
        #
        #     if arearatio < 27:
        #         cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #         hand_gesture = 3
        #     else:
        #         cv2.putText(frame, 'ok', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #         hand_gesture = 6
        #
        # elif l == 4:
        #     cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #     hand_gesture = 4
        #     timer = +1
        #
        # elif l == 5:
        #     cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #     hand_gesture = 5
        #     timer=+1
        #
        # elif l == 6:
        #     cv2.putText(frame, 'reposition!', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        #     hand_gesture = 7
        #
        # else:
        #     cv2.putText(frame, 'reposition', (10, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        # show the windows

        # start = text_for_detection(hand_gesture)
        cv2.imshow('mask', mask)
        # cv2.imshow('frame', frame)




    except:
        pass

    k = cv2.waitKey(100) & 0xFF
    if k == 27 or k == ord('q'):
        break



# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
