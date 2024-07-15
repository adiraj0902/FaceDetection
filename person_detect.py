import cv2
import tensorflow as tf


# Load a pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
      
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        
        cv2.putText(frame, "Person ", (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    if len(faces) == 0:
        
        cv2.putText(frame, "Nothing detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
