import cv2
import imutils
import numpy as np
import dlib
from mtcnn import MTCNN  # published in 2016 by Zhang et al.
from facenet_pytorch import MTCNN as facenet_MTCNN  # published in 2015 by Google researchers Schroff et al
import torch

# Source webcam = 0 or file path
source = "data/video.avi"

# Initializing Detectors
# Haar
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
# DNN
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
# Dlib
dlib_hog_svm_detector = dlib.get_frontal_face_detector()
# Dlib Cnn
dlib_cnn_face_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
# MTCNN - consists of 3 neural networks connected in a cascade.
mtcnn_detector = MTCNN()
# Google MTCNN - uses embeddings.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
facenet_mtcnn = facenet_MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(source)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = imutils.resize(frame, width=320)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # type CV_8U containing an image for Haar
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # must be 8bit gray or RGB image for Dlib

        Haar_frame = frame.copy()
        DNN_frame = frame.copy()
        Dlib_hog_frame = frame.copy()
        Dlib_cnn_frame = frame.copy()
        mtcnn_frame = frame.copy()
        facenet_mtcnn_frame = frame.copy()

        # HaarCascade
        faces = faceCascade.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        for i, (x, y, w, h) in enumerate(faces[0]):
            conf = faces[1][i]
            if conf > 10:
                # text = f"{conf:.2f}%"
                text = "Opencv_Haar"
                cv2.putText(Haar_frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 1)
                cv2.rectangle(Haar_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

        # DNN
        (h, w) = DNN_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(DNN_frame, (100, 100)), 1.0, (100, 100), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # text = "{:.2f}%".format(conf * 100)
                text = "DNN"
                cv2.putText(DNN_frame, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
                cv2.rectangle(DNN_frame, (startX, startY), (endX, endY), (0, 255, 0), 3)

        # Dlib Hog_SVM
        faces = dlib_hog_svm_detector(rgb, 1)  # result
        # to draw faces on image
        for result in faces:
            startX = result.left()
            startY = result.top()
            endX = result.right()
            endY = result.bottom()
            text = "Dlib_Hog_Svm"
            cv2.putText(Dlib_hog_frame, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 1)
            cv2.rectangle(Dlib_hog_frame, (startX, startY), (endX, endY), (255, 0, 0), 3)

        # Dlib CNN
        faces_cnn = dlib_cnn_face_detector(rgb, 1)  # result
        # to draw faces on image
        for result in faces_cnn:
            startX = result.rect.left()
            startY = result.rect.top()
            endX = result.rect.right()
            endY = result.rect.bottom()
            text = "Dlib_CNN"
            cv2.putText(Dlib_cnn_frame, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 1)
            cv2.rectangle(Dlib_cnn_frame, (startX, startY), (endX, endY), (255, 0, 0), 3)

        # MTCNN
        boxes = mtcnn_detector.detect_faces(mtcnn_frame)
        if boxes:
            box = boxes[0]['box']
            conf = boxes[0]['confidence']
            x, y, w, h = box[0], box[1], box[2], box[3]
            if conf > 0.5:
                text = "MTCNN"
                cv2.rectangle(mtcnn_frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
                cv2.putText(mtcnn_frame, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 0), 1)

        # FaceNet - MTCNN
        boxes, conf = facenet_mtcnn.detect(facenet_mtcnn_frame)
        if conf[0]:
            for (x, y, w, h) in boxes:
                # text = f"{conf[0] * 100:.2f}%"
                text = "FaceNet"
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.putText(facenet_mtcnn_frame, text, (x, y - 20),  cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1)
                cv2.rectangle(facenet_mtcnn_frame, (x, y), (w, h), (0, 255, 255), 1)

        h1 = cv2.hconcat([Haar_frame, DNN_frame,Dlib_hog_frame])
        h2 = cv2.hconcat([mtcnn_frame, facenet_mtcnn_frame, Dlib_cnn_frame])
        fin = cv2.vconcat([h1, h2])
        #cv2.imshow("Opencv_Haar", Haar_frame)
        #cv2.imshow("DNN", DNN_frame)
        #cv2.imshow("Dlib_hog", Dlib_hog_frame)
        #cv2.imshow("Dlib_cnn", Dlib_cnn_frame)
        #cv2.imshow("Mtcnn", mtcnn_frame)
        #cv2.imshow("Facenet", facenet_mtcnn_frame)
        cv2.imshow("All Detectors", fin)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
