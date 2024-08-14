import cv2
import numpy as np
from models.inference import predict_mood

def capture_and_predict():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for prediction
        img = cv2.resize(frame, (64, 64))
        img = img / 255.0
        # No need to expand dimensions here, the model expects (64, 64, 3)
        # img = np.expand_dims(img, axis=0)  # Remove this line

        mood = predict_mood(img)

        # Display the frame with mood
        cv2.putText(frame, f'Mood: {mood}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Mood Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_and_predict()
