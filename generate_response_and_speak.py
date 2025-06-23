
import edge_tts
from playsound import playsound
from dotenv import load_dotenv
from playsound import playsound
from openai import OpenAI
import os
from fastapi.responses import StreamingResponse
import string
from langdetect import detect # Or use lingua-py, fasttext

# Load environment and OpenAI client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
PUNCTUATION_TRANSLATOR = str.maketrans('', '', string.punctuation)

# --------------------------
# Speech: Edge TTS speaking
# --------------------------
async def speak_edge(text):
    try:
        detected_lang_code = detect(text)
        print(f"Detected language: {detected_lang_code}")

        # --- IMPORTANT: You need a mapping from lang code to edge_tts voice ---
        # This is a simplified example. You'd ideally get a list of
        # available voices and pick the best match, or have a pre-defined mapping.
        voice_mapping = {
            "en": "en-US-JennyNeural",
            "ur": "ur-PK-UzmaNeural",
            "fr": "fr-FR-DeniseNeural",
            "hi": "hi-IN-MadhurNeural"
            # Add more mappings as needed
        }

        voice_name = voice_mapping.get(detected_lang_code)

        if not voice_name:
            print(f"No suitable Edge TTS voice found for language: {detected_lang_code}. Defaulting to English.")
            voice_name = "en-IN-NeerjaNeural" # Fallback

        communicate = edge_tts.Communicate(text, voice_name)
        await communicate.save("response.mp3")
        # os.system("afplay response.mp3")  # macOS-safe playback
        # Play a local audio file
        playsound('response.mp3')
        print(f"Speech saved to output_auto.mp3 using voice: {voice_name}")

    except Exception as e:
        print(f"An error occurred: {e}")

# --------------------------
# Streaming GPT Response + Speak
# --------------------------
async def generate_response_and_speak(prompt: str):
    full_response = ""
    chunk_buffer = ""
    sentence_endings = (".", "!", "?")
    messages = [{"role": "user", "content": prompt}]

    async def stream():
        nonlocal full_response, chunk_buffer

        for chunk in client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                chunk_buffer += token
                yield f"data: {token}\n\n"  # SSE token streaming

                # Speak full sentences
                if any(chunk_buffer.endswith(e) for e in sentence_endings) and len(chunk_buffer.strip().split()) > 10:
                    await speak_edge(chunk_buffer.strip())
                    chunk_buffer = ""

        # Speak remaining tail
        if chunk_buffer.strip():
            await speak_edge(chunk_buffer.strip())
        print(full_response)

        yield f"data: [END]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")