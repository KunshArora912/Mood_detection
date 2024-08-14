import tensorflow as tf
import numpy as np

def load_model():
    return tf.keras.models.load_model('models/mood_detection_model.h5')

def preprocess_image(img):
    # Resize the image to the target size
    img = tf.image.resize(img, [64, 64])
    # Normalize pixel values to [0, 1]
    img = img / 255.0
    # Expand dimensions to match model input
    img = np.expand_dims(img, axis=0)
    return img

def predict_mood(img):
    model = load_model()
    # Check if the input shape is as expected
    if img.shape != (64, 64, 3):
        raise ValueError(f"Expected input shape (64, 64, 3), but got {img.shape}")

    # Predict mood
    prediction = model.predict(np.expand_dims(img, axis=0))  # Add batch dimension for prediction

    # Assuming the model outputs a probability distribution
    mood = np.argmax(prediction, axis=-1)[0]

    return mood

if __name__ == '__main__':
    import sys
    img_path = sys.argv[1]
    print(predict_mood(img_path))
