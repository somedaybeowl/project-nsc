import os
import json
import time
import subprocess
from queue import Queue
from threading import Thread
import pickle
import audioop
import playsound

import pyaudio
from vosk import Model, KaldiRecognizer

# --- Config ---
MODEL_PATH = "/Users/manutsanan/Downloads/speech to text/vosk-inference/model"

CHANNELS = 1
FRAME_RATE = 16000
CHUNK = 1000
RECORD_SECONDS = 1000000
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_SIZE = 2
INPUT_DEVICE_INDEX = 0

with open('bully_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('bully_model.pkl', 'rb') as f:
    bully_model = pickle.load(f)

messages = Queue()
recordings = Queue()

def record_microphone():
    print("Start")
    p = pyaudio.PyAudio()
    stream = p.open(format=AUDIO_FORMAT,
                    channels=CHANNELS,
                    rate=FRAME_RATE,
                    input=True,
                    input_device_index=INPUT_DEVICE_INDEX,
                    frames_per_buffer=CHUNK)

    frames = []
    while not messages.empty():
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        print(".", end="", flush=True)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("\n Stop")

    recordings.put(frames)

model = Model(MODEL_PATH)
recognizer = KaldiRecognizer(model, FRAME_RATE)
recognizer.SetWords(True)

def speech_recognition():
    print("ถอดเสียง")
    frames = recordings.get()
    audio_data = b''.join(frames)

    threshold = 500
    for i in range(0, len(audio_data), CHUNK):
        chunk_data = audio_data[i:i+CHUNK]

        rms = audioop.rms(chunk_data, SAMPLE_SIZE)
        if rms > threshold:
            recognizer.AcceptWaveform(chunk_data) 
        else:
            pass

    final_result = recognizer.FinalResult()
    text = json.loads(final_result).get("text", "")
    print("[FINAL TEXT]", text)

    if text.strip() != "":
        X_test = vectorizer.transform([text])
        X_test_dense = X_test.toarray()
        prediction = bully_model.predict(X_test_dense)[0]
        proba = bully_model.predict_proba(X_test_dense)[0]

        print(f"[BULLY PREDICTION] {prediction}")
        print(f"[PROBABILITIES] {proba}")

        if prediction == 0:
            print("Bully")
            playsound.playsound('notify.MP4')
        else:
            print("None")


def start_recording():
    messages.put(True)
    record_thread = Thread(target=record_microphone)
    transcribe_thread = Thread(target=speech_recognition)

    record_thread.start()
    input("กด Enter เพื่อหยุดอัดเสียง")
    messages.get()
    record_thread.join()

    transcribe_thread.start()
    transcribe_thread.join()

if __name__ == "__main__":
    input("กด Enter เพื่อเริ่มอัดเสียง")
    start_recording()



