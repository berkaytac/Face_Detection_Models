import cv2
import dlib
import imutils
import numpy as np
from mtcnn import MTCNN
from facenet_pytorch import MTCNN as face_MTCNN
import torch


def opencv_haar(source):
    faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)
        for i, (x, y, w, h) in enumerate(faces[0]):
            conf = faces[1][i]
            if conf > 10:
                text = f"{conf:.2f}%"
                cv2.putText(frame, text, (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def opencv_dnn(source):
    modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
    configFile = "models/deploy.prototxt.txt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (100, 100)), 1.0, (100, 100), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]
            if conf > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                text = "{:.2f}%".format(conf * 100)
                cv2.putText(frame, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 3)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def dlib_hog_svm_face_detector(source):
    detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb, 1)  # result
        # to draw faces on image
        for result in faces:
            startX = result.left()
            startY = result.top()
            endX = result.right()
            endY = result.bottom()
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 3)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def dlib_cnn(source):
    detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector(rgb, 1)  # result
        # to draw faces on image
        for result in faces:
            startX = result.rect.left()
            startY = result.rect.top()
            endX = result.rect.right()
            endY = result.rect.bottom()
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 3)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def mtcnn_face_detector(source):
    detector = MTCNN()
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        boxes = detector.detect_faces(frame)
        if boxes:
            box = boxes[0]['box']
            conf = boxes[0]['confidence']
            x, y, w, h = box[0], box[1], box[2], box[3]
            if conf > 0.5:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def facenet_face_detector(source):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Create the model
    mtcnn = face_MTCNN(keep_all=True, device=device)
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=640)
        boxes, conf = mtcnn.detect(frame)
        if conf[0]:
            for (x, y, w, h) in boxes:
                text = f"{conf[0] * 100:.2f}%"
                x, y, w, h = int(x), int(y), int(w), int(h)
                cv2.putText(frame, text, (x, y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
            cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# source = 0
source = "data.avi"
opencv_haar(source)
# opencv_dnn(source)
# dlib_hog_svm_face_detector(source)
# dlib_cnn(source)
# mtcnn_face_detector(source)
# facenet_face_detector(source)
