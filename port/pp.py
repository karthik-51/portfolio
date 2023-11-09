import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained emotion detection model (you can download one online)
emotion_model = load_model('emotion_model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar Cascade for face detection (you can download one online)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) for the detected face
        roi_gray = gray[y:y+h, x:x+w]
        
        # Resize the ROI to match the input size expected by the emotion detection model
        roi_gray = cv2.resize(roi_gray, (48, 48))
        
        # Normalize the ROI
        roi_gray = roi_gray / 255.0
        
        # Reshape the ROI to fit the model input shape (1, 48, 48, 1)
        roi_gray = np.reshape(roi_gray, (1, 48, 48, 1))
        
        # Predict the emotion using the model
        predicted_emotion = emotion_model.predict(roi_gray)
        
        # Get the index of the predicted emotion
        emotion_index = np.argmax(predicted_emotion)
        
        # Get the corresponding emotion label
        emotion = emotion_labels[emotion_index]
        
        # Draw a rectangle around the detected face and label the emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Display the frame with emotion labels
    cv2.imshow('Emotion Detection', frame)
    
    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
