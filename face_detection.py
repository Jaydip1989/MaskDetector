import os
import cv2
import tensorflow
from tensorflow import keras
import numpy as np


model = keras.models.load_model('FaceMaskDetector.hdf5')
face_clfs = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
source = cv2.VideoCapture(0)

labels_dict = {0: 'with mask', 1: 'without mask'}
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}

while(True):
    ret, img = source.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_clfs.detectMultiScale(img, 1.25, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+h]
        resized = cv2.resize(face_img, (224, 224))
        normalized = resized/255.0

        reshaped = np.reshape(normalized, (1, 224, 224, 3))
        result = model.predict(reshaped)

        label = np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(img, (x, y), (x+w, y+h), color_dict[label], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), color_dict[label], -2)
        cv2.putText(img, labels_dict[label], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow("LIVE", img)
    key = cv2.waitKey(3)
    if (key == 1):
        break

source.release()
cv2.destroyAllWindows()