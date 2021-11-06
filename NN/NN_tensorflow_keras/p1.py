import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import mnist

# https://victorzhou.com/blog/keras-neural-network-tutorial/
# https://victorzhou.com/blog/intro-to-cnns-part-1/

train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

model = Sequential([
    Dense(64, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

model.load_weights('model.h5')

'''
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=5,
    batch_size=32,
)


model.evaluate(
    test_images,
    to_categorical(test_labels)
)
'''

predictions = model.predict(test_images[:5])

print(np.argmax(predictions, axis=1))

print(test_labels[:5])