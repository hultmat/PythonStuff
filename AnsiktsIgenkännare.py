import numpy as np
import cv2 as cv
import pickle
import dlib
from PIL import Image
import os
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist


face_cascade = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

# Fram till rad 60ish innehåller skriptet hur den läser in ansikten. 
# Progammet fungerar snabbare när man har de i ett skillt skript men för att underlätta inlämningen baka jag ihop allting
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "ansikten")

recognizer = cv.face.LBPHFaceRecognizer_create()

current_id = 0
label_ids ={}

y_labels = []
x_train = []


#läser in bilderna och skapar arrays av dem samt ger en id&label whatiswhat
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path)).lower()
            #print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1

            id_ =label_ids[label]

            pil_image = Image.open(path).convert('L')
            size = (500, 500)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, 'uint8')
            
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

print(label_ids)

with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save('trainner.yml') 

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {'personName': 1}
with open('labels.pickle', 'rb') as f:
    origlabels = pickle.load(f)
    labels = {v: k for k, v in origlabels.items()}


cap = cv.VideoCapture(0)

def eye_aspect_ratio(eye):

	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	C = dist.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 40
COUNTER = 0
BLINKNINGAR = 0

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


while(True):

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=800)
    #frame = cv.resize(frame, (0, 0), fx=0.7, fy=0.7)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)


    for rect in rects:
        shape=predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye= shape[lStart:lEnd]
        rightEye= shape[rStart:rEnd]
        leftEar= eye_aspect_ratio(leftEye)
        rightEar= eye_aspect_ratio(rightEye)

        ear= (leftEar+rightEar)/2.0
        

        leftEyeHull=cv.convexHull(leftEye)
        rightEyeHull= cv.convexHull(rightEye)

        cv.drawContours(frame, [leftEyeHull], -1, (255,255, 255),1)
        cv.drawContours(frame, [rightEyeHull], -1, (255,255, 255),1)

        


    for (x, y, w, h) in faces:
        # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        if ear < EYE_AR_THRESH:
            COUNTER +=1
            BLINKNINGAR +=0.25
            
            cv.putText(frame, "Blinkning", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            #print('blinkning')

            if COUNTER > EYE_AR_CONSEC_FRAMES:
                cv.putText(frame, "Sover du?", (150, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                #print('soverdu?')
        else:
            COUNTER=0
        
        #cv.putText(frame, '{:.2f} EAR'.format(ear), (w,h),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),2)

    
        #cv.putText(frame, 'Blinkningar: {:1.0f}'.format(BLINKNINGAR), (10, 80), cv.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

        ids, conf = recognizer.predict(roi_gray)

        if conf >= 40:
            # print(ids)
            #print(labels[ids])
            cv.putText(frame, labels[ids] + '  {:.2f} EAR'.format(ear), (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)

        width = x + w
        height = y + h
        cv.rectangle(frame, (x, y), (width, height), (150, 0, 0), 2)

    cv.imshow('WebCam', frame)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
