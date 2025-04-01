import numpy as np
import librosa
import tensorflow as tf
import sounddevice as sd
import wave
import os
import time
import soundfile as sf

# Configuration Parameters
TARGET_SR = 16000         
NUM_MFCC = 13              
MAXLEN = 400               
N_FFT = 2048               
HOP_LENGTH = 512           
CLASS_NAMES = ['Maju', 'Mundur', 'Kanan', 'Kiri', 'Berhenti'] 

TRAIN_MEAN_PATH = '/Users/naufalhaekall/Documents/UNIVERSITY/Academic Documents/Skripsi/Code/models/MFCC/train_mean.npy'       
TRAIN_STD_PATH = '/Users/naufalhaekall/Documents/UNIVERSITY/Academic Documents/Skripsi/Code/models/MFCC/train_std.npy'         
MODEL_PATH = '/Users/naufalhaekall/Documents/UNIVERSITY/Academic Documents/Skripsi/Code/models/MFCC/lstmModel_earlyStop.h5'    
RECORDINGS_DIR = './recordings/'  

# Ensure recordings directory exists
os.makedirs(RECORDINGS_DIR, exist_ok=True)

# Model and Parameters
model = tf.keras.models.load_model(MODEL_PATH)
mean = np.load(TRAIN_MEAN_PATH)
std = np.load(TRAIN_STD_PATH)

print("Model and normalization parameters loaded successfully.")

# Audio Recording Function
def record_audio(filename="output.wav", duration=5, sample_rate=16000):
    print("Recording... Press Ctrl+C to stop early.")

    try:
        # Record audio with the given settings
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,  # Mono
            dtype="int16"
        )
        sd.wait()  # Wait until the recording is finished
        print("Recording finished. Saving audio...")

        # Save the recorded audio to a WAV file
        with wave.open(filename, "wb") as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data.tobytes())
        print(f"Audio saved as {filename}")

    except KeyboardInterrupt:
        print("Recording interrupted. No audio saved.")

# Feature Extraction and Prediction
def extract_features(signal, sr=16000, num_mfcc=13, n_fft=2048, hop_length=512, maxlen=400):
    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        mfcc_features = np.vstack([mfccs, delta_mfccs, delta2_mfccs]).T
        if mfcc_features.shape[0] > maxlen:
            mfcc_features_padded = mfcc_features[:maxlen, :]
        elif mfcc_features.shape[0] < maxlen:
            padding = np.zeros((maxlen - mfcc_features.shape[0], mfcc_features.shape[1]))
            mfcc_features_padded = np.vstack([mfcc_features, padding])
        else:
            mfcc_features_padded = mfcc_features
        return mfcc_features_padded
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.zeros((maxlen, num_mfcc * 3))

def make_prediction(features, mean, std, model, class_names):
    features_norm = (features - mean) / std
    features_norm = np.expand_dims(features_norm, axis=0)
    prediction = model.predict(features_norm)
    predicted_index = np.argmax(prediction, axis=1)[0]
    return class_names[predicted_index] if predicted_index < len(class_names) else "Unknown"

# Prediction App
class AudioPredictionApp:
    def __init__(self):
        pass

    def run(self, duration=5):
        print("Starting prediction app...")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        wav_filename = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

        # Record audio
        record_audio(filename=wav_filename, duration=duration, sample_rate=TARGET_SR)

        # Load the recorded audio
        signal, sr = librosa.load(wav_filename, sr=TARGET_SR)

        # Extract features
        features = extract_features(signal, sr=TARGET_SR, num_mfcc=NUM_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, maxlen=MAXLEN)

        # Make a prediction
        predicted_class = make_prediction(features, mean, std, model, CLASS_NAMES)
        print(f"[PREDICTION] Predicted Class: {predicted_class}")

if __name__ == "__main__":
    app = AudioPredictionApp()
    app.run(duration=3)