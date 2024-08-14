import cv2
import os
import numpy as np

def preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def process_directory(input_dir, output_dir, target_size=(64, 64)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_file)
        img = preprocess_image(img_path, target_size)
        img_name = os.path.basename(img_path)
        output_path = os.path.join(output_dir, img_name)
        np.save(output_path, img)

def main():
    moods = ['happy', 'sad', 'angry', 'neutral']
    for mood in moods:
        input_dir = f'data/raw/{mood}'
        output_dir = f'data/processed/{mood}'
        process_directory(input_dir, output_dir)

if __name__ == '__main__':
    main()
