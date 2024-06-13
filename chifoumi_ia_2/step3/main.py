import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

NUMBER_TRY = 14

TRAINING_DIR = "../rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')


VALIDATION_DIR = "../rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)
train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
    batch_size=126
)

model = tf.keras.models.Sequential([
    # the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    # The second convolution
    # The third convolution
    # The fourth convolution
    # Flatten the results to feed into a DNN
    # 512 neuron hidden layer
])

#compile model with loss as categorical_crossentropy, optimizer as rmsprop and metrics as accuracy

#train the model and test his validity
history = model.fit(...)

model.save("my_model.h5")


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()