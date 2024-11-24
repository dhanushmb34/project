import os
import librosa
import numpy as np
import speech_recognition as sr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D # type: ignore

# Dataset path and emotion labels
DATA_PATH = "audio_dataset/"
emotion_labels = {"neutral": 0, "happy": 1, "sad": 2, "angry": 3, "fearful": 4, "surprised": 5}

# Function to extract MFCC features
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare dataset
def prepare_dataset():
    X, y = [], []
    for label, value in emotion_labels.items():
        folder = os.path.join(DATA_PATH, label)
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".wav"):
                    features = extract_features(os.path.join(folder, file))
                    if features is not None:
                        X.append(features)
                        y.append(value)
        else:
            print(f"Folder not found: {folder}")
    return np.array(X), np.array(y)

# Build the emotion recognition model
def build_emotion_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv1D(128, 5, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train and save emotion model
def train_emotion_model():
    print("Preparing the dataset...")
    X, y = prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=-1)  # Shape: (samples, 40, 1)
    X_test = np.expand_dims(X_test, axis=-1)

    print("Building the emotion model...")
    model = build_emotion_model(input_shape=(40, 1), num_classes=len(emotion_labels))

    print("Training the emotion model...")
    history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    os.makedirs("models", exist_ok=True)
    model.save("models/emotion_model.h5")
    print("Emotion model saved successfully!")

# Predict emotion from audio
def predict_emotion(audio_path, model):
    features = extract_features(audio_path)
    if features is not None:
        features = np.expand_dims(features, axis=(0, -1))  # Shape: (1, 40, 1)
        prediction = model.predict(features)
        emotion_index = np.argmax(prediction)
        emotion = [key for key, value in emotion_labels.items() if value == emotion_index][0]
        return emotion
    else:
        print("Could not extract features for emotion prediction.")
        return None

# Recognize speech and predict emotion using the same audio
def process_audio(audio_path, model):
    recognizer = sr.Recognizer()
    try:
        # Step 1: Use audio file for speech recognition
        print("Recognizing speech...")
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio)
            print(f"Recognized Text: {recognized_text}")

        # Step 2: Predict emotion from the same audio file
        print("Analyzing emotion...")
        predicted_emotion = predict_emotion(audio_path, model)
        print(f"Predicted Emotion: {predicted_emotion}")

        return recognized_text, predicted_emotion
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None, None

# Analyze context with GPT-2 (instead of GPT-3.5)
def analyze_with_gpt2(text, emotion):
    print("Sending context to GPT-2 for analysis...")
    try:
        # Load GPT-2 model and tokenizer
        model_name = "gpt2"  # You can use "gpt2-medium" or "gpt2-large" if desired
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)

        # Prepare input text
        input_text = f"The following text was spoken with a {emotion} emotion: '{text}'. Provide insights about the context and emotion."
        inputs = tokenizer(input_text, return_tensors="pt")

        # Generate response
        outputs = model.generate(inputs['input_ids'], max_length=150)
        gpt2_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"GPT-2 Analysis: {gpt2_response}")
    except Exception as e:
        print(f"Error during GPT-2 analysis: {e}")

# Main function
if __name__ == "__main__":
    print("Starting the system...")

    # Train the emotion recognition model if it doesn't exist
    if not os.path.exists("models/emotion_model.h5"):
        train_emotion_model()

    # Load the emotion model
    print("Loading the emotion model...")
    emotion_model = load_model("models/emotion_model.h5")
    print("Emotion model loaded successfully!")

    # Record and process audio
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording audio...")
        try:
            audio = recognizer.listen(source, timeout=5)
            with open("temp_audio.wav", "wb") as f:
                f.write(audio.get_wav_data())
            print("Audio recorded successfully!")

            # Process the recorded audio
            recognized_text, predicted_emotion = process_audio("temp_audio.wav", emotion_model)

            # Analyze context with GPT-2
            if recognized_text and predicted_emotion:
                analyze_with_gpt2(recognized_text, predicted_emotion)
        except Exception as e:
            print(f"Error during recording: {e}")
