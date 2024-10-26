import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import load_data, preprocess_data

if __name__ == "__main__":
    images, labels = load_data('data/test')
    images, labels = preprocess_data(images, labels)

    model = load_model('model.h5')
    test_loss, test_acc = model.evaluate(images, labels)
    print(f'Test accuracy: {test_acc:.2f}')
