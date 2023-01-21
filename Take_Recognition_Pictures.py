# import the opencv library
import cv2
face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./data/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('./data/haarcascade_smile.xml')

# define a video capture object
face_id = input("enter the id")
count=0
vid = cv2.VideoCapture(0)

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #Added Lines
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #increment counter

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)

        # print("voila", count)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 25)

        for (xs, ys, ws, sh) in smile:
            cv2.rectangle(roi_color, (xs, ys), (xs + ws, ys + sh), (255, 0, 0), 1)
        count += 1;
        cv2.imwrite("./dataset/User." + str(face_id) + '.' +
                    str(count) + ".jpg", gray[y:y + h, x:x + w])
    #Finished added lines

    # Display the resulting frame
    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    cv2.imshow('Face recognition', frame)


    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    """ if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
    """
    if k == 27:
        break;

    elif count >= 30:  # Take 30 face sample and stop video
        break;

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()