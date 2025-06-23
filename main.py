from fastapi import FastAPI
from record_voice_with_microphone import record_voice_with_microphone
from transcribe_audio import transcribe_audio
from generate_response_and_speak import generate_response_and_speak
app = FastAPI()

# --------------------------
# FastAPI Route
# --------------------------
@app.get("/voice-chat")
async def voice_chat():
    audio_path = record_voice_with_microphone()
    question = transcribe_audio(audio_path)
    print(f"📝 Transcription: {question}")
    return await generate_response_and_speak(question)