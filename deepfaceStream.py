import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import time
import re
import os
from deepface import DeepFace
from deepface.extendedmodels import Age
from deepface.commons import functions, realtime, distance as dst
from deepface.detectors import FaceDetector
from request import sendTags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def reworked_stream(model_name='VGG-Face', detector_backend='opencv', distance_metric='cosine',
                    enable_face_analysis=True, source=0, time_threshold=5, frame_threshold=1,display_frame = False):
    '''
        param model_name:  one of face analysis models
        param detector_backend: one of face detection models
        param distance_metric: cosine, euclidian
        param enable_face_analysis:
        param source:
        param time_threshold: how many second analyzed image will be displayed
        param frame_threshold: how many frames required to focus on face
        param display_frame : display the analyzed video stream ,false to production moode

        as the result labels are sent to the api and printed to the console
        return: none
    '''
    report = []
    face_detector = FaceDetector.build_model(detector_backend)
    print("Detector backend is ", detector_backend)

    if enable_face_analysis == True:
        tic = time.time()
        age_model = DeepFace.build_model('Age')
        print("Age model loaded")

        gender_model = DeepFace.build_model('Gender')
        print("Gender model loaded")

        toc = time.time()

        print("Facial attribute analysis models loaded in ", toc - tic, " seconds")
    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequentially 5 frames
    freezed_frame = 0
    ticer= time.time()

    cap = cv2.VideoCapture(source)  # webcam

    timer = True
    # remove c-r
    counter = 0
    while(timer):
        ret, img = cap.read()

        if img is None:
            break
        # next step is to detect faces on a screen
        # find a replacement
        #print('---- something ----')
        raw_img = img.copy()
        resolution = img.shape; resolution_x = img.shape[1]; resolution_y = img.shape[0]

        try:
            # faces store list of detected_face and region pair
            faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align=False)
        except:  # to avoid exception if no face detected
            faces = []
        if len(faces) == 0:
            face_included_frames = 0

        detected_faces = []
        face_index = 0

        for face, (x, y, w, h) in faces:
            #print(f' face is {type(face)} \n, (x,y,w,h) coord {x,y,w,h} \n')
            if w > 130:
                face_detected = True
                if face_index == 0:
                    face_included_frames = face_included_frames + 1
                # ---------------------
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]  # crop detected face
                detected_faces.append((x, y, w, h))
                face_index = face_index + 1
                # ---------------------
        #print(f'face_detected {face_detected} freeze {freeze} ,\n face included frames {face_included_frames}  frame threshold { frame_threshold}')
        if face_detected == True and (face_included_frames % frame_threshold)==0 and freeze == False:
            freeze = True
            # base_img = img.copy()
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()
            tic = time.time()

        if freeze == True:
            toc = time.time()
            #if(toc-tic) < time_threshold:
            if freezed_frame == 0:
                freeze_img = base_img.copy()
               # print(f' len of detected faces {detected_faces_final}')
                for detected_face in detected_faces_final:
                    x = detected_face[0];
                    y = detected_face[1]
                    w = detected_face[2];
                    h = detected_face[3]

                    custom_face = base_img[y:y + h, x:x + w]
                    # --------------
                    if enable_face_analysis == True:
                        counter = counter+1
                        print(f'facial analysis + count {counter}')
                        gray_img = functions.preprocess_face(img=custom_face, target_size=(48, 48), grayscale=True,
                                                             enforce_detection=False, detector_backend='opencv')
                        face_224 = functions.preprocess_face(img=custom_face, target_size=(224, 224),
                                                             grayscale=False, enforce_detection=False,
                                                             detector_backend='opencv')

                        age_predictions = age_model.predict(face_224)[0, :]
                        apparent_age = Age.findApparentAge(age_predictions)

                        # -------------------------------

                        gender_prediction = gender_model.predict(face_224)[0, :]

                        if np.argmax(gender_prediction) == 0:
                            gender = "W"
                        elif np.argmax(gender_prediction) == 1:
                            gender = "M"
                        # -------------
                        color = "\033[94m" if gender == "M" else "\033[93m"
                        endC ="\033[0m"
                        sendTags(apparent_age, gender)
                        analysis_report = str(int(apparent_age)) + " " + gender
                        report.append((apparent_age, gender))
                        print(f'{color} analysis_report \n {analysis_report} {endC}')
            '''
            time_left = int(time_threshold - (toc - tic) + 1)
            cv2.imshow('img', freeze_img)
            freezed_frame = freezed_frame + 1
            '''
            face_detected = False
            face_included_frames = 0
            freeze = False
            freezed_frame = 0

        if display_frame:
            cv2.imshow('img', img)
        #exit condition not working
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            break
        ''' 
        if time.time()-ticer <= 40000:
            pass
            #timer = False
        '''
    for i in report:
        print(i,'\n')
    vals =[]
    gend=[]
    for it in report:
        (i,j)=it
        vals.append(i)
        gend.append(j)

    print(f'avarage predicted is {sum(vals)/len(report)}  actual 29')
    print(f'precision score gender pred {gend.count("W")/len(report)}')
    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()
    print(f'took time {time.time()-ticer}')