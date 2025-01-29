import sounddevice as sd
import numpy as np
from vosk import Model, KaldiRecognizer
import json
from datetime import datetime

# Initialize the Vosk speech recognition model
MODEL_PATH = "vosk-model-small-en-us-0.15"  # Replace with the path to your Vosk model
try:
    model = Model(MODEL_PATH)
    recognizer = KaldiRecognizer(model, 16000)
except Exception as e:
    print(f"Error loading Vosk model: {e}")
    exit()

# Parameters for audio analysis
SAMPLERATE = 16000  # 16 kHz sampling rate
VOLUME_THRESHOLD = 0.3  # Adjust based on ambient noise levels
HIGH_VOLUME_DURATION = 2  # Seconds of sustained high volume to trigger an alert
MONITOR_DURATION = 10  # Duration to check audio in seconds (optional if you want a time limit)

# Logs for speech and high volume events
speech_log = []
high_volume_log = []

# Function to process audio stream
def audio_callback(indata, frames, time, status):
    global high_volume_start

    if status:
        print(f"Audio status error: {status}")
        return

    # Calculate RMS (Root Mean Square) for volume level
    rms = np.sqrt(np.mean(np.square(indata)))

    # Check for high volume
    if rms > VOLUME_THRESHOLD:
        if high_volume_start is None:
            high_volume_start = datetime.now()
    else:
        if high_volume_start is not None:
            duration = (datetime.now() - high_volume_start).total_seconds()
            if duration >= HIGH_VOLUME_DURATION:
                log_entry = f"High volume detected from {high_volume_start.strftime('%Y-%m-%d %H:%M:%S')} to {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."
                high_volume_log.append(log_entry)
                print(log_entry)
            high_volume_start = None

    # Convert audio to int16 and feed it to the recognizer
    audio_data = (indata * 32767).astype(np.int16).tobytes()
    if recognizer.AcceptWaveform(audio_data):
        result = json.loads(recognizer.Result())
        if result.get("text"):
            speech_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Detected speech: {result['text']}")
            print(f"Speech detected: {result['text']}")

# Initialize variables
high_volume_start = None

# Start audio stream
print("Starting audio monitoring... Press Ctrl+C to stop.")
try:
    with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=audio_callback):
        sd.sleep(MONITOR_DURATION * 1000 if MONITOR_DURATION else 100000)  # Monitor for specified time or indefinitely
except KeyboardInterrupt:
    print("Audio monitoring stopped.")
except Exception as e:
    print(f"Error during audio monitoring: {e}")

# Save logs to files
with open("speech_log.txt", "w") as speech_file:
    speech_file.write("Speech Log\n")
    speech_file.write("===================\n")
    for entry in speech_log:
        speech_file.write(f"{entry}\n")

with open("high_volume_log.txt", "w") as volume_file:
    volume_file.write("High Volume Log\n")
    volume_file.write("===================\n")
    for entry in high_volume_log:
        volume_file.write(f"{entry}\n")

print("Logs saved: speech_log.txt and high_volume_log.txt.")
