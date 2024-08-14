import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

def preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image at path {img_path} could not be loaded.")
    else:
        img = cv2.resize(img, target_size)
        img = img / 255.0
    return img


def load_data():
    # Adjust the path as necessary
    base_path = 'data/processed'
    categories = ['angry', 'happy', 'neutral', 'sad']
    images = []
    labels = []

    for label, category in enumerate(categories):
        category_path = os.path.join(base_path, category)
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            try:
                # Check if file is a .npy file and load it
                if file_path.endswith('.npy'):
                    image = np.load(file_path)
                    images.append(image)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image at path {file_path}: {e}")

    images = np.array(images)
    labels = np.array(labels)

    # Split into training and validation datasets (example split)
    from sklearn.model_selection import train_test_split
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2,
                                                                          random_state=42)

    return train_images, train_labels, val_images, val_labels