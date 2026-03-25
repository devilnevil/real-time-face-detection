import cv2
face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
webcam = cv2.VideoCapture(0)
while True:
    status, video = webcam.read()

    gray = cv2.cvtColor(video, cv2.COLOR_RGB2GRAY)

    face = face_model.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in face:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 200, 0), 3)

    cv2.imshow("Display", video)
    if cv2.waitKey(1) == ord("q"):
        break
webcam.release()
cv2.destroyAllWindows()
