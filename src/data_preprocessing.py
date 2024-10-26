import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images = []
    labels = []
    classes = os.listdir(data_dir)

    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (224, 224))  # Resize to match model input
            images.append(img)
            labels.append(class_name)

    return np.array(images), np.array(labels)

def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0  # Normalize to [0, 1]
    return images, labels

if __name__ == "__main__":
    images, labels = load_data('data/train')
    images, labels = preprocess_data(images, labels)
