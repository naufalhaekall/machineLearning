import tensorflow as tf
from google.colab import files
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_image(path):
  # OpenCV to read image
  image = cv2.imread(path)
  # Handle potential errors 
  if image is None:
    raise ValueError(f"Error loading image: {path}")
  print(f"Loaded image with shape: {image.shape}")
  return image

def preprocess_image(image):
  # Resize image 
  resized_image = cv2.resize(image, (224, 224))  
  print(f"Resized image shape: {resized_image.shape}")
  # Convert to HSV colorspace
  hsv_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2HSV)
  print(f"Converted to HSV with shape: {hsv_image.shape}")
  # Normalize pixel values 
  normalized_image = hsv_image / 255.0 
  print(f"Normalized image with pixel values between 0 and 1")

  return normalized_image

def extract_skin_features(image):
    # Skin color model in RGB space
    lower_skin = np.array([0, 48, 80], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    # Skin pixels using skin color model
    mask = cv2.inRange(image, lower_skin, upper_skin)
    # Percentage of skin pixels in the image
    skin_pixels = np.count_nonzero(mask)
    total_pixels = image.shape[0] * image.shape[1]
    skin_percentage = skin_pixels / total_pixels
    # Extracted features: skin percentage
    features = np.array([skin_percentage])
    print(f"Extracted features: {features}")
    return features

# Example usage (assuming you have the image path)
image_path = "/content/drive/MyDrive/Academic/Data/Test/brownTest/10.jpg"

try:
  image = load_image(image_path)
  preprocessed_image = preprocess_image(image)
  skin_features = extract_skin_features(preprocessed_image)
except ValueError as e:
  print(f"Error: {e}")


def load_data(data_dir, label_file, cnn_model_path):
    images = []
    labels = []

    # Load pre-trained CNN model
    cnn_model = load_trained_cnn_model(cnn_model_path)
    # Read labels from file
    with open(label_file, 'r') as f:
        for line in f:
            image_path, label = line.strip().split(';')
            labels.append(label)
    # Load and preprocess images
    for image_path in data_dir:
        if os.path.isdir(image_path):
            # Iterate over all image files in the directory
            for filename in os.listdir(image_path):
                if filename.endswith('.jpg') or filename.endswith('.png'):
                    full_image_path = os.path.join(image_path, filename)
                    image = load_image(full_image_path)
                    preprocessed_image = preprocess_image(image)
                    features = extract_skin_features(preprocessed_image)
                    images.append(features)
        else:
            # Load a single image
            image = load_image(image_path)
            preprocessed_image = preprocess_image(image)
            features = extract_skin_features(preprocessed_image)
            images.append(features)

    # Convert data to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels, cnn_model

def load_trained_cnn_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Preprocess input images
def preprocess_input_image(image):
    image = cv2.resize(image, (224, 224))
    return image

# Map predicted makeup shades to recommendations
def map_to_recommendation(prediction):
    # Define your mapping from predicted makeup shades to recommendations
    num_classes = 10  # Update this with the actual number of classes
    # Define a list of brand recommendations
    brand_recommendations = ["Beauty Bakerie", "Covergirl", "bareMinerals", "Estee Lauder", "Revlon","Maybelline","L'Oreal","Black Up","Elsas Pro"]
    # Assuming num_classes is the number of classes predicted by your model
    num_classes = len(brand_recommendations)
    # Construct recommendations list using brand recommendations
    recommendations = [brand_recommendations[i] for i in range(num_classes)]
    return recommendations[prediction]

# Classify makeup shades and provide recommendations
def classify_and_recommend(image_paths):
    recommendations = []
    for image_path in image_paths:
        # Load and preprocess the input image
        image = preprocess_input_image(load_image(image_path))
        # Make predictions using the loaded model
        prediction = np.argmax(cnn_model.predict(np.expand_dims(image, axis=0)))
        print("Prediction:", prediction)  # Debugging statement
        # Map predicted makeup shade to recommendation
        recommendation = map_to_recommendation(prediction)
        recommendations.append(recommendation)
    return recommendations

# Example usage:
data_dir = ['/content/drive/MyDrive/Academic/Data/Test/brownTest']  # Your data directory
label_file = '/content/drive/MyDrive/Academic/Data/brownShades.csv'  # Your label file
cnn_model_path = '/content/drive/MyDrive/Academic/Data/makeUpRecommendationModel.h5'  # Path to the pre-trained CNN model

# Load data and model
images, labels, cnn_model = load_data(data_dir, label_file, cnn_model_path)

# Classify and recommend for input images
image_paths = ['//content/24284023.jpg', '/content/37846614.jpg']
recommendations = classify_and_recommend(image_paths)

# Output recommendations
for image_path, recommendation in zip(image_paths, recommendations):
    print(f"Image: {image_path}, Recommendation: {recommendation}")