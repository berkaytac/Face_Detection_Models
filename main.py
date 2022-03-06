import cv2
import os
import imutils
import numpy as np
import dlib
from mtcnn import MTCNN  # published in 2016 by Zhang et al.
from facenet_pytorch import MTCNN as facenet_MTCNN  # published in 2015 by Google researchers Schroff et al
import torch

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


def draw_text(frame, text, col):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    text_color_bg = (255, 255, 255)
    text_pos = (10, 10)
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(frame, text_pos, (text_pos[0] + text_w, text_pos[1] + text_h), text_color_bg, -1)
    cv2.putText(frame, text, (text_pos[0], text_pos[1] + text_h), font, font_scale, col, font_thickness)
    return frame


def draw_face_rect(frame, x, y, w, h, col):
    cv2.rectangle(frame, (x, y), (x + w, y + h), col, 1)
    return frame


def opencv_haar(frame, gray):
    '''
    type CV_8U containing an image for Haar
    very well explained in here: https://medium.com/analytics-vidhya/haar-cascades-explained-38210e57970d
    works with gray image.
    '''
    Haar_frame = frame.copy()
    faces = faceCascade.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)
    text = "Opencv_Haar"
    col = (0, 0, 255)
    Haar_frame = draw_text(Haar_frame, text, col)
    for i, (x, y, w, h) in enumerate(faces[0]):
        conf = faces[1][i]
        if conf > 10:
            Haar_frame = draw_face_rect(Haar_frame, x, y, w, h, col)
    return Haar_frame


def dnn_face(frame):
    """
    OpenCV’s deep neural network module with Caffe model
    The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
    The .caffemodel file which contains the weights for the actual layers
    OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework with a ResNet base network
    """
    DNN_frame = frame.copy()
    (frameHeight, frameWidth) = DNN_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(DNN_frame, (100, 100)), 1.0, (100, 100), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    text = "DNN"
    col = (0, 255, 0)
    DNN_frame = draw_text(DNN_frame, text, col)
    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > 0.15:
            box = detections[0, 0, i, 3:7] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
            (x, y, endX, endY) = box.astype("int")
            w = endX - x
            h = endY - y
            DNN_frame = draw_face_rect(DNN_frame, x, y, w, h, col)
    return DNN_frame


def dlib_hog(frame, rgb, dlib_hog_svm_detector):
    """
    It requires RGB or GRAY image.
    HOG + Linear SVM: dlib.get_frontal_face_detector()
    MMOD CNN: dlib.cnn_face_detection_model_v1(modelPath)
    """
    Dlib_hog_frame = frame.copy()
    faces = dlib_hog_svm_detector(rgb, 1)  # result
    text = "Dlib_Hog_Svm"
    col = (255, 0, 0)
    Dlib_hog_frame = draw_text(Dlib_hog_frame, text, col)
    # to draw faces on image
    for result in faces:
        x = result.left()
        y = result.top()
        endX = result.right()
        endY = result.bottom()
        w = endX - x
        h = endY - y
        Dlib_hog_frame = draw_face_rect(Dlib_hog_frame, x, y, w, h, col)
    return Dlib_hog_frame


def dlib_cnn(frame, rgb, dlib_cnn_face_detector):
    """
    It requires RGB or GRAY image.
    MMOD CNN: dlib.cnn_face_detection_model_v1(modelPath)
    """
    Dlib_cnn_frame = frame.copy()
    faces_cnn = dlib_cnn_face_detector(rgb, 1)  # result
    text = "Dlib_CNN"
    col = (255, 0, 255)
    Dlib_cnn_frame = draw_text(Dlib_cnn_frame, text, col)
    # to draw faces on image
    for result in faces_cnn:
        x = result.rect.left()
        y = result.rect.top()
        endX = result.rect.right()
        endY = result.rect.bottom()
        w = endX - x
        h = endY - y
        Dlib_cnn_frame = draw_face_rect(Dlib_cnn_frame, x, y, w, h, col)
    return Dlib_cnn_frame


def mtcnn_face(frame):
    # https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/
    mtcnn_frame = frame.copy()
    boxes = mtcnn_detector.detect_faces(mtcnn_frame)
    text = "MTCNN"
    col = (255, 255, 0)
    mtcnn_frame = draw_text(mtcnn_frame, text, col)
    if boxes:
        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        if conf > 0.5:
            mtcnn_frame = draw_face_rect(mtcnn_frame, x, y, w, h, col)
    return mtcnn_frame


def facenet_mtcnn_detector(frame):
    # https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/
    facenet_mtcnn_frame = frame.copy()
    boxes, conf = facenet_mtcnn.detect(facenet_mtcnn_frame)
    text = "FaceNet"
    col = (0, 110, 255)
    facenet_mtcnn_frame = draw_text(facenet_mtcnn_frame, text, col)
    if conf[0]:
        for (x, y, endx, endy) in boxes:
            x, y, endx, endy = int(x), int(y), int(endx), int(endy)
            w = endx - x
            h = endy - y
            facenet_mtcnn_frame = draw_face_rect(facenet_mtcnn_frame, x, y, w, h, col)
    return facenet_mtcnn_frame


def video_face_detector(source):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=240)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Opencv Haar
            Haar_frame = opencv_haar(frame, gray)
            # DNN
            DNN_frame = dnn_face(frame)
            # Dlib Hog_Svm
            Dlib_hog_frame = dlib_hog(frame, rgb, dlib_hog_svm_detector)
            # Dlib CNN
            Dlib_cnn_frame = dlib_cnn(frame, rgb, dlib_cnn_face_detector)
            # MTCNN
            mtcnn_frame = mtcnn_face(frame)
            # FaceNet - MTCNN
            facenet_mtcnn_frame = facenet_mtcnn_detector(frame)

        # cv2.imshow("Haar_frame", Haar_frame)
        # cv2.imshow("DNN_frame", DNN_frame)
        # cv2.imshow("Dlib_hog_frame", Dlib_hog_frame)
        # cv2.imshow("Dlib_CNN", Dlib_cnn_frame)
        # cv2.imshow("mtcnn_frame", mtcnn_frame)
        # cv2.imshow("facenet_mtcnn_frame", facenet_mtcnn_frame)
        h1 = cv2.hconcat([Haar_frame, DNN_frame, Dlib_hog_frame])
        h2 = cv2.hconcat([mtcnn_frame, facenet_mtcnn_frame, Dlib_cnn_frame])
        fin = cv2.vconcat([h1, h2])
        cv2.imshow("All Detectors", fin)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def image_face_detector(source):
    frame = cv2.imread(source)
    frame = imutils.resize(frame, width=480)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Opencv Haar
    Haar_frame = opencv_haar(frame, gray)
    # DNN
    DNN_frame = dnn_face(frame)
    # Dlib Hog_Svm
    Dlib_hog_frame = dlib_hog(frame, rgb, dlib_hog_svm_detector)
    # Dlib CNN
    Dlib_cnn_frame = dlib_cnn(frame, rgb, dlib_cnn_face_detector)
    # MTCNN
    mtcnn_frame = mtcnn_face(frame)
    # FaceNet - MTCNN
    facenet_mtcnn_frame = facenet_mtcnn_detector(frame)

    # cv2.imshow("Haar_frame", Haar_frame)
    # cv2.imshow("DNN_frame", DNN_frame)
    # cv2.imshow("Dlib_hog_frame", Dlib_hog_frame)
    # cv2.imshow("Dlib_CNN", Dlib_cnn_frame)
    # cv2.imshow("mtcnn_frame", mtcnn_frame)
    # cv2.imshow("facenet_mtcnn_frame", facenet_mtcnn_frame)
    h1 = cv2.hconcat([Haar_frame, DNN_frame, Dlib_hog_frame])
    h2 = cv2.hconcat([mtcnn_frame, facenet_mtcnn_frame, Dlib_cnn_frame])
    fin = cv2.vconcat([h1, h2])
    cv2.imshow("All Detectors", fin)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(base_dir, "data")
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            path = os.path.join(root, file)
            if file.lower().endswith((".avi", ".mp4", ".mov")):
                video_face_detector(path)
            if file.lower().endswith((".jpg", ".png", ".bmp",".heif",".jpeg")):
                image_face_detector(path)
