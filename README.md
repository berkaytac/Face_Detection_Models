The purpose of this side project is to see if any technique can perfrom better face detection than Dlib's HOG base detector for the purpose of Heart Rate Monitoring.
The results show that the Caffe model of OpenCV's dnn module performs best.

## Installation

- Install these dependencies (imutils, Numpy, Dlib, Mtcnn, Torch, Facenet-pytorch, Opencv-Python):

```
pip install -r requirements.txt
```

> The Dlib library has four primary prerequisites: Boost, Boost.Python, CMake and X11/XQuartx. [Read this article](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) to know how to easily install them.

- Place picture or video under the [data folder](https://github.com/berkaytac/Face_Detection_Models/tree/main/data). 

Run the Face Detection Models:

```
python main.py
```

*video_face_detector_combined.py and video_face_detector_models.py are some initial work files that can be ignored. 

# Face Detection Models

Implementation of face detection models.
- [Open CV - Haar](#haar)
- [Open CV - DNN](#dnn)
- [Dlib - HOG + SVM](#hog)
- [Dlib - CNN](#cnn)
- [MTCNN](#mtcnn)
- [FACENET - MTCNN](#facenet)

## Open CV - Haar <a name="haar">

- very well explained in here: https://medium.com/analytics-vidhya/haar-cascades-explained-38210e57970d
- works with gray image (type CV_8U containing an image for Haar)

```python
import cv2
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for i, (x, y, w, h) in enumerate(faces[0]):
        conf = faces[1][i]
        if conf > 10:
            text = f"{conf:.2f}%"
            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)
```

## Open CV - DNN <a name="dnn">

-  OpenCV’s deep neural network module with Caffe model 
-  The .prototxt file(s) which define the model architecture (i.e., the layers themselves)
-  The .caffemodel file which contains the weights for the actual layers
-  OpenCV’s deep learning face detector is based on the Single Shot Detector (SSD) framework with a ResNet base network

```python
import cv2
modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
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
```
## Dlib - HOG + SVM <a name="hog">
- It requires RGB or GRAY image.
- HOG + Linear SVM: dlib.get_frontal_face_detector()

```python
import cv2
import dlib
dlib_hog_svm_detector = dlib.get_frontal_face_detector()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = dlib_hog_svm_detector(rgb, 1)  # result
    for result in faces:
        startX = result.left()
        startY = result.top()
        endX = result.right()
        endY = result.bottom()
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 3)
```
## Dlib - CNN <a name="cnn">
- It requires RGB or GRAY image.
- MMOD CNN: dlib.cnn_face_detection_model_v1(modelPath)

```python
import cv2
import dlib
dlib_hog_svm_detector = dlib.get_frontal_face_detector()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = dlib_hog_svm_detector(rgb, 1)  # result
    for result in faces:
        startX = result.rect.left()
        startY = result.rect.top()
        endX = result.rect.right()
        endY = result.rect.bottom()
        cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 3)
```

## MTCNN <a name="mtcnn">
- Published in 2016 by Zhang et al.
- MTCNN - consists of 3 neural networks connected in a cascade.
- https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/

```python
import cv2
from mtcnn import MTCNN
mtcnn_detector = MTCNN()
mtcnn_frame = frame.copy()
detector = MTCNN()
    boxes = detector.detect_faces(frame)
    if boxes:
        box = boxes[0]['box']
        conf = boxes[0]['confidence']
        x, y, w, h = box[0], box[1], box[2], box[3]
        if conf > 0.5:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
```

## FACENET - MTCNN <a name="facenet">
- published in 2015 by Google researchers Schroff et al
- https://arsfutura.com/magazine/face-recognition-with-facenet-and-mtcnn/
- Google MTCNN - uses embeddings

```python
import cv2
import torch
from facenet_pytorch import MTCNN as facenet_MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
facenet_mtcnn = facenet_MTCNN(keep_all=True, device=device)
    boxes, conf = facenet_mtcnn.detect(facenet_mtcnn_frame)
    if conf[0]:
        for (x, y, w, h) in boxes:
            text = f"{conf[0] * 100:.2f}%"
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (170, 170, 170), 1)
        cv2.rectangle(frame, (x, y), (w, h), (255, 255, 255), 1)
```
