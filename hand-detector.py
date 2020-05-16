'''
Written by Abbosjon Kudratov -u1610001
for Multimedi Computing Course Project
@Inha University in Tashkent

May16, 2020

Note: some parts of the code were taken 
from here - https://github.com/Sadaival/Hand-Gestures  and were used for reference

'''


import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import math




#declare global variables
kernel = np.ones((3, 3), np.uint8)
font = cv2.FONT_HERSHEY_DUPLEX


'''
some other comment goes here

'''





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

    find_defects(defects,approx,roi,areacnt, arearatio, frame) #calling other function by passing arguments





def find_defects(defects, approx, roi,areacnt, arearatio, frame):
    l=0 # defect count
    
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
    text_for_detection(l+1,areacnt,arearatio,frame)   # Note that we need to call with l+1
  




def process_image(img):
    if img is not None:
        # define region of interest
        roi = frame[100:500, 100:500]
        

        cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), 2) #draw green rectangle to detect gestures inside
        
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

    
    '''
    hand gestures:
    0=empty
    1=one
    2=two
    3=three or  OKAY
    4=four
    5=hello/palm
    6=other/reposition 
    '''

    texts = ['Reposition!',
             'Put your hand in the box',
             'Like',
             'OKAY',
             '5 - hello/palm']
    
    if l == 1:
        if areacnt < 2000:
            cv2.putText(frame, texts[1], (0, 50), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
        else:
            if arearatio < 12:
                cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            elif arearatio < 17.5:
                cv2.putText(frame, texts[2], (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

            else:
                cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
             

    elif l == 2:
        cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
      

    elif l == 3:

        if arearatio < 27:
            cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            
        else:
            cv2.putText(frame, texts[3], (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            

    elif l == 4:
        cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
     

    elif l == 5:
        cv2.putText(frame, texts[4], (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
     
      

    elif l == 6:
        cv2.putText(frame, texts[0], (0, 50), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
        

    else:
        cv2.putText(frame, texts[0], (10, 50), font, 2, (255, 0, 0), 3, cv2.LINE_AA)

    #show the resulting frame 
    cv2.imshow('frame', frame)
   






cap = cv2.VideoCapture(0)
while True:

    try:  # an error comes if it does not find anything in window as it cannot find contour of max area
        # therefore this try error statement

        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h,w,d = frame.shape
  
        roi = frame[100:500, 100:500] #defining ROI to detect gestures
       
        mask = process_image(frame)

        contours_convex_hull(mask,roi,frame)



        # show the masking window
        cv2.imshow('mask', mask)
       



    except:
        pass

    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break



# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
