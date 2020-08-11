from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
from time import time
from keras.models import load_model
from keras.applications import VGG16

import urllib.request
EMOTION_DICT = {2:"ANGRY", 1:"FEAR", 3:"HAPPY", 0:"NEUTRAL", 4:"SAD", 5:"SURPRISE"}
model_VGG = VGG16(weights='imagenet', include_top=False)
model_top = load_model("Data/Model_Save/model.h5")

def make_prediction(img):
    
    cv2.imwrite("test.jpg",img)
    img=cv2.imread("test.jpg")
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    if len(faces)==1:
        flag=0
        for (x,y,w,h) in faces:
            face_clip = img[y:y+h, x:x+w]
            img=cv2.resize(face_clip, (48,48))
            flag=1
        if flag==0:
            img=cv2.resize(img,(48,48))
        cv2.imwrite("test1.jpg",img)
        read_image = img
        read_image = read_image.reshape(1, read_image.shape[0], read_image.shape[1], read_image.shape[2])
        read_image_final = read_image/255.0  #normalizing the image
        VGG_Pred = model_VGG.predict(read_image_final)  #creating bottleneck features of image using VGG-16.
        
        VGG_Pred = VGG_Pred.reshape(1, VGG_Pred.shape[1]*VGG_Pred.shape[2]*VGG_Pred.shape[3])
        top_pred = model_top.predict(VGG_Pred)  #making prediction from our own model.
        emotion_label = top_pred[0].argmax()
        print("Predicted Expression Probabilities")
        print("ANGRY: {}\nFEAR: {}\nHAPPY: {}\nNEUTRAL: {}\nSAD: {}\nSURPRISE:{}\n".format(top_pred[0][2], top_pred[0][1], top_pred[0][3], top_pred[0][0], top_pred[0][4],top_pred[0][5]))
        
        return EMOTION_DICT[emotion_label],max(top_pred[0])
    else:
        str1="No Face Detected"
        str2="None"
        return str1,str2

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def width_height_ratio(leftbrow,rightbrow,jaw,mouth,nose):
    midbrow=[]
    midbrow.append((leftbrow[0][0]+rightbrow[4][0])//2)
    midbrow.append((leftbrow[0][1]+rightbrow[4][1])//2)
    horizontal=dist.euclidean(jaw[1], jaw[15])
    vertical=dist.euclidean(midbrow,mouth[3])
    whr=(horizontal/vertical)
    return whr
    
    
    
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor",default='C:/Users/kumar/Desktop/MINOR_2/expression/shape_predictor_68_face_landmarks.dat')

args = vars(ap.parse_args())

EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 1
wink_frames=1
COUNTER = 0
TOTAL = 0
wink=0
cwink=0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(jstart,jend) =face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(rbrowstart,rbrowend) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
(lbrowstart,lbrowend) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
(mouthstart,mouthend) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(nosestart,noseend) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]


print("[INFO] starting video stream thread...")
#fileStream = True
#vs = VideoStream(src=0).start()

#fileStream = False

url="http://192.168.1.5:5656/shot.jpg?rnd=364547"

start=time()
bcounter=0
wcount=0
abnor=0
nor=0
lap=time()
present=time()
emotion=eyeact="default"
whr=0.1234
while True:
    
    #if fileStream and not vs.more():
        #break
    #frame = vs.read()
    #frame = imutils.resize(frame, width=640)

    imgresp=urllib.request.urlopen(url)
    imgnp=np.array(bytearray(imgresp.read()),dtype=np.uint8)
    frame=cv2.imdecode(imgnp,-1)
    frame = imutils.resize(frame, width=720)
    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    now=time()
    emoflag=0
    if now-present>5:
        emotion,valu=make_prediction(gray)
        emoflag=1
        present=now
    rects = detector(gray, 0)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftbrow =shape[lbrowstart:lbrowend]
        rightbrow = shape[rbrowstart:rbrowend]
        jaw = shape[jstart:jend]
        mouth = shape[mouthstart:mouthend]
        nose= shape[nosestart:noseend]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if leftEAR < EYE_AR_THRESH and rightEAR<EYE_AR_THRESH:
            COUNTER += 1
        elif leftEAR<EYE_AR_THRESH or rightEAR<EYE_AR_THRESH:
            wink+=1
        else:
            if COUNTER >= EYE_AR_CONSEC_FRAMES and COUNTER<=5:
                TOTAL += 1
            if wink>=wink_frames and wink<=6:
                cwink+=1
            COUNTER = 0
            wink=0
        
        ntime=time()
        flag=0
        
        if TOTAL-bcounter>0:
            flag=1
        
            if (TOTAL-bcounter)/(ntime-start)>1:
                abnor+=1
                bsus="Abnormal"
                start=ntime
                bcounter=TOTAL
                wcount=cwink
            else:
                nor+=1
                bsus="Normal"
                start=ntime
                bcounter=TOTAL
            if cwink-wcount>0:
            
             flag=1
             bsus="Abnormal"
             wcount=cwink
             abnor+=1
        if lap+5<ntime:
        
            if nor==0:
                nor=1
            if abnor/nor>0.5:
                #print("Eye Activity : Abnormal")
                eyeact="Abnormal"
            else:
                #print("Eye Activity : Normal")
                eyeact="Normal"
            abnor=0
            nor=0
            lap=ntime
            whr=width_height_ratio(leftbrow,rightbrow,jaw,mouth,nose)
            
        cv2.putText(frame, "Randomness: {}".format(cwink), (200, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (17, 163, 255), 2)
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (400, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #cv2.putText(frame, "Blink Activity: {}".format(bsus), (10, 90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "WHR: {:.3f}".format(whr), (250, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (17, 163, 255), 2)
        cv2.putText(frame, "Eye Activity: {}".format(eyeact), (10, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "Dominant Emotion: {}".format(emotion), (400, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 2, (1, 1,255), -1)
    #if emoflag==1:
        #print("Dominant Emotion:{} Value:{}\n\n".format(emotion,valu))
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
cv2.destroyAllWindows()
#vs.stop()                  
                    
                         
        

        
    
