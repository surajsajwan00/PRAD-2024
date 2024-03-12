## Libraries
import cv2
import numpy as np
from keras.models import load_model

## Load the model
model = load_model('gesture_model.h5')

## Video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (256, 256))
    frame = frame.reshape(1, 256, 256, 1)
    frame = frame/255.0

    # Predicting the gesture amongst[fist, five, none, okay, peace, rad, straight, thumbs]
    prediction = model.predict(frame)
    gesture = np.argmax(prediction)

    # Display the resulting frame with predicted gesture[fist, five, none, okay, peace, rad, straight, thumbs]
    cv2.putText(frame, "Predicted Gesture: "+ str(gesture), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hand Gesture Recognition', frame)

    # 'q' key to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

## Release and destroy the capture window
cap.release()
cv2.destroyAllWindows()