import tensorflow as tf
import numpy as np
import gzip, os

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = f"{BASE_DIR}/model"
DATA_DIR = f"{BASE_DIR}/data"

def load_local_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28, 28)

def load_local_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

train_images = load_local_mnist_images(f'{DATA_DIR}/train-images-idx3-ubyte.gz')
train_labels = load_local_mnist_labels(f'{DATA_DIR}/train-labels-idx1-ubyte.gz')

# Load test images and labels from local files
test_images = load_local_mnist_images(f'{DATA_DIR}/t10k-images-idx3-ubyte.gz')
test_labels = load_local_mnist_labels(f'{DATA_DIR}/t10k-labels-idx1-ubyte.gz')


train_images = train_images / 255.0

test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=20)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

model.save(MODEL_DIR)

