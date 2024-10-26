import numpy as np
import cv2
from tensorflow.keras.models import load_model

def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    prediction = model.predict(img)
    class_index = np.argmax(prediction[0])
    return class_index

if __name__ == "__main__":
    model = load_model('model.h5')
    image_path = 'path/to/your/image.jpg'
    class_index = predict_image(model, image_path)
    print(f'Predicted class index: {class_index}')
