import tensorflow as tf
from tensorflow.keras import layers, models
from google.colab import files

def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    # Flatten layer
    model.add(layers.Flatten())
    # Dense layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    # Compile 
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Example usage:
input_shape = (224, 224, 3)  # Adjust input image size
num_classes = 15  # Adjust number of makeup shades
model = create_cnn_model(input_shape, num_classes)
model.summary()

# Save and Download model
# model.save('makeUpRecommendationModel.h5')
# files.download('makeUpRecommendationModel.h5')