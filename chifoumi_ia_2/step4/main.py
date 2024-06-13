import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
import numpy as np

model = tf.keras.models.load_model('my_model.h5')

if __name__ == "__main__":
    while(1):
        img = "../rps-test-set/scissors/testscissors01-03.png"
        img = image.load_img(path, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        result = classes.argmax()
        if result == 0:
            print("it's a paper!")
        if result == 1:
            print("it's a rock!")
        if result == 2:
            print("it's a scissor!")