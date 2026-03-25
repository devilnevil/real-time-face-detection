import cv2

face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

webcam = cv2.VideoCapture(0)

while True:
    status, video = webcam.read()

    if not status:
        break  # safety

    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
    face = face_model.detectMultiScale(gray)

    for (x, y, w, h) in face:
        cv2.rectangle(video, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Display", video)

    # 👇 KEYBOARD CONTROL
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
