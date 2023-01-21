import time

import cv2
import numpy as np
import os
import pandas as pd
import time
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
#cascadePath = "haarcascade_frontalface_default.xml"

faceCascade =cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
#font = cv2.FONT_HERSHEY_PLAIN
# iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
#names = ['none','Marouane', 'Obama','ismail ','karima']
new_df = pd.read_csv('./data/persons.csv')
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, img = cam.read()
    #img = cv2.flip(img, -1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        timing=time.time()
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        print("done in % 0.3fs" % (time.time()-timing))
        subid=''
        # If confidence is less them 100 ==> "0" : perfect match
        if (confidence < 70):

            df_res = new_df.loc[new_df['id'] == int(id)];
            print(df_res.to_numpy()[0])
            arr = df_res.to_numpy()[0]
            id = arr[1]
            subid=arr[2]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "inconnu"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            str(id),
            (x + 3, y - 5),
            font,
            0.7,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(subid),
            (x + 5, y + h+20),
            font,
            0.5,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            str(confidence),
            (x + 5, y + h - 5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
    if k == 27:
        break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()