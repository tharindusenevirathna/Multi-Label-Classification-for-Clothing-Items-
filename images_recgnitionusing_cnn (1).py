# Import necessary libraries for building and training the CNN model
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Number of classes in the dataset
NUM_CLASSES = 10

# Reshaped size for the input images (28*28 pixels to a flat array)
RESHAPED = 784

# Number of neurons in hidden layers
HIDDEN_NEURONS = 256

# Dimensions of the input image
IMAGE_ROWS, IMAGE_COLS = 28, 28

# Shape of the input for the model
INPUT_SHAPE = (IMAGE_ROWS, IMAGE_COLS, 1)

# Load the Fashion MNIST dataset
fashion_data = tf.keras.datasets.fashion_mnist

# Split the data into training and testing sets
(train_images, train_labels), (test_images, test_labels) = fashion_data.load_data()

# Reshape the images to add a channel dimension (for grayscale images)
train_images = train_images.reshape(60000, IMAGE_ROWS, IMAGE_COLS, 1)
test_images = test_images.reshape(10000, IMAGE_ROWS, IMAGE_COLS, 1)

# Normalize the images to the range [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Dictionary to map label indices to class names
labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot"
}

# Plot some sample images from the training set
plt.figure(figsize=(10, 10))
for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i].reshape((IMAGE_ROWS, IMAGE_COLS)))
    label_index = train_labels[i]
    plt.title(labels[label_index])

plt.show()

# Convert labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
test_labels = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)

# Define a simple CNN model for the Fashion MNIST dataset
class cnn_fmnist:
    @staticmethod
    def build(input_shape, classes):
        # Sequential model to add layers in order
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPooling2D(pool_size=(2, 2)),

            Flatten(),

            Dense(1024, name='dense_layer1', activation='relu'),
            Dense(512, name='dense_layer2', activation='relu'),
            Dense(NUM_CLASSES, name='output_layer', activation='softmax')
        ])
        return model

# Build the CNN model
model = cnn_fmnist.build(input_shape=INPUT_SHAPE, classes=NUM_CLASSES)

# Compile the model with Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display model architecture
model.summary()

# Define training parameters
BATCH_SIZE = 128
EPOCHS = 50
VALIDATION_SPLIT = 0.2

# Train the model with the training data
history = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, verbose=1)

# Evaluate the model with the test data
score = model.evaluate(test_images, test_labels)
print('\nTest score; ', score[0])
print('Test Accuracy: ', score[1])

# Plot training and validation loss over epochs
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('Train-Loss Function')

# Plot training and validation accuracy over epochs
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Train-Accuracy')

# Predict classes for the test images
predicted_classes = np.around(model.predict(test_images))
predicted_classes = np.argmax(predicted_classes, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Find correctly and incorrectly classified images
correct = np.nonzero(predicted_classes == test_labels)[0]
incorrect = np.nonzero(predicted_classes != test_labels)[0]

# Plot some correctly classified images
plt.figure(figsize=(15, 15))
for i, indx in enumerate(correct[:16]):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[indx].reshape((IMAGE_ROWS, IMAGE_COLS)), cmap="Greens")
    plt.title("True:{} Pred:{}".format(labels[test_labels[indx]], labels[predicted_classes[indx]]))

plt.show()

# Plot some incorrectly classified images
plt.figure(figsize=(15, 15))
for i, indx in enumerate(incorrect[:16]):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[indx].reshape((IMAGE_ROWS, IMAGE_COLS)), cmap="Reds")
    plt.title("True:{} Pred:{}".format(labels[test_labels[indx]], labels[predicted_classes[indx]]))

plt.show()

# Define a new CNN model with additional regularization (Dropout and L2 regularization)
class cnn_fmnist_new:
    @staticmethod
    def build(input_shape, classes, use_l2_reg=False, l2_loss_lambda=0.0025):
        l2 = regularizers.l2(l2_loss_lambda) if use_l2_reg else None
        if l2 is not None:
            print('Using L2 regularization with lambda=%.6f' % l2_loss_lambda)

        # Build model with additional regularization layers
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2, input_shape=input_shape),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.15),

            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.2),

            Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=l2),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),

            Flatten(),
            Dropout(0.4),

            Dense(1024, name='dense_layer1', activation='relu', kernel_regularizer=l2),
            Dropout(0.4),

            Dense(512, name='dense_layer2', activation='relu', kernel_regularizer=l2),
            Dropout(0.2),

            Dense(NUM_CLASSES, name='output_layer', activation='softmax')
        ])
        return model

# Build the new CNN model with L2 regularization and Dropout
model = cnn_fmnist_new.build(input_shape=INPUT_SHAPE, classes=NUM_CLASSES, use_l2_reg=True)

# Compile the model with Adam optimizer and lower learning rate
adam = Adam(learning_rate=0.0001)
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the new model architecture
model.summary()

# Define new training parameters for the modified model
BATCH_SIZE = 128
EPOCHS = 100
VALIDATION_SPLIT = 0.2

# Train the modified model with the training data
history = model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=VALIDATION_SPLIT, verbose=1)

# Predict classes for the test images using the modified model
predicted_classes = np.around(model.predict(test_images))
predicted_classes = np.argmax(predicted_classes, axis=1)
test_labels = np.argmax(test_labels, axis=1)

# Find correctly and incorrectly classified images for the modified model
correct = np.nonzero(predicted_classes == test_labels)[0]
incorrect = np.nonzero(predicted_classes != test_labels)[0]

# Plot some correctly classified images for the modified model
plt.figure(figsize=(15, 15))
for i, indx in enumerate(correct[:16]):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[indx].reshape((IMAGE_ROWS, IMAGE_COLS)), cmap="Greens")
    plt.title("True:{} Pred:{}".format(labels[test_labels[indx]], labels[predicted_classes[indx]]))

plt.show()

# Plot some incorrectly classified images for the modified model
plt.figure(figsize=(15, 15))
for i, indx in enumerate(incorrect[:16]):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[indx].reshape((IMAGE_ROWS, IMAGE_COLS)), cmap="Reds")
    plt.title("True:{} Pred:{}".format(labels[test_labels[indx]], labels[predicted_classes[indx]]))

plt.show()

# Plot training and validation loss over epochs for the modified model
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.title('Train-Loss Function')

# Plot training and validation accuracy over epochs for the modified model
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Train-Accuracy')

# Convert test labels back to one-hot encoded format for evaluation
from keras.utils import to_categorical
test_labels = to_categorical(test_labels, num_classes=10)

# Evaluate the modified model with the test data
score = model.evaluate(test_images, test_labels)
print('\nTest score; ', score[0])
print('Test Accuracy: ', score[1])
