
import tempfile
import speech_recognition as sr
# --------------------------
# Microphone Audio Recorder
# --------------------------
def record_voice_with_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        print("🎙️ Recording... Speak now.")
        audio_data = recognizer.listen(source, phrase_time_limit=20)
        print("✅ Recording complete.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_data.get_wav_data())
        return f.name