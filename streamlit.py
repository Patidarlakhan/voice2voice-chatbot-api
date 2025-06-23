
# import streamlit as st
# import sounddevice as sd
# import numpy as np
# import whisper
# import tempfile
# import time
# from langchain.llms import HuggingFacePipeline
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from transformers import pipeline

# # Load Whisper Model (small for better speed)
# whisper_model = whisper.load_model("base")

# # Load Local HuggingFace Model (like LLaMA, Mistral, Falcon)
# @st.cache_resource
# def load_llm():
#     pipe = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", max_new_tokens=100, temperature=0.7)
#     return HuggingFacePipeline(pipeline=pipe)

# llm = load_llm()

# # LangChain Setup
# prompt = PromptTemplate.from_template("Q: {question}\nA:")
# chain = LLMChain(llm=llm, prompt=prompt)

# # Record Voice
# def record_audio(duration=5, fs=16000):
#     st.info("Recording...")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
#     st.success("Recording complete!")
#     return audio, fs

# def save_audio_to_file(audio, fs):
#     with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
#         from scipy.io.wavfile import write
#         write(tmpfile.name, fs, audio)
#         return tmpfile.name

# def transcribe(audio_path):
#     return whisper_model.transcribe(audio_path)["text"]

# # Streamlit UI
# st.title("🎙️ Voice to LangChain Chatbot (Local)")

# if st.button("🎤 Record Voice"):
#     audio, fs = record_audio(duration=5)
#     audio_path = save_audio_to_file(audio, fs)

#     st.audio(audio_path)

#     st.subheader("📝 Transcription:")
#     text = transcribe(audio_path)
#     st.write(text)

#     st.subheader("🤖 Streaming LLM Response:")
#     # Simulated streaming
#     response = chain.run(text)
#     response_placeholder = st.empty()
#     full_response = ""
#     for word in response.split():
#         full_response += word + " "
#         response_placeholder.markdown(f"**{full_response.strip()}**")
#         time.sleep(0.1)










# import streamlit as st
# import sounddevice as sd
# import numpy as np
# import tempfile
# import openai
# import os
# from scipy.io.wavfile import write
# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()
# client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# st.set_page_config(page_title="🎤 Voice to Chatbot", layout="centered")
# st.title("🎙️ Voice-to-Text + LangChain Streaming (OpenAI)")

# # Record audio
# def record_audio(duration=5, fs=16000):
#     st.info("Recording...")
#     audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
#     sd.wait()
#     st.success("Recording complete!")
#     return audio, fs

# # Save audio to temp .wav file
# def save_audio(audio, fs):
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
#         write(tmpfile.name, fs, audio)
#         return tmpfile.name

# # Transcribe audio using OpenAI Whisper API
# def transcribe_audio(file_path):
#     with open(file_path, "rb") as audio_file:
#         transcript = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file
#         )
#     return transcript.text

# # Stream response using OpenAI ChatCompletion
# def stream_chat_response(prompt):
#     messages = [{"role": "user", "content": prompt}]
#     response_container = st.empty()
#     full_response = ""

#     for chunk in client.chat.completions.create(
#         model="gpt-4",
#         messages=messages,
#         stream=True
#     ):
#         if 'choices' in chunk and len(chunk['choices']) > 0:
#             delta = chunk['choices'][0]['delta']
#             if 'content' in delta:
#                 full_response += delta['content']
#                 response_container.markdown(full_response + "▌")

#     response_container.markdown(full_response)
#     return full_response

# # Main button
# if st.button("🎤 Record & Ask"):
#     audio, fs = record_audio()
#     path = save_audio(audio, fs)

#     st.audio(path)
#     st.subheader("📝 Transcription")
#     question = transcribe_audio(path)
#     st.success(f"User said: **{question}**")

#     st.subheader("🤖 AI Response")
#     stream_chat_response(question)



import streamlit as st
import os
import tempfile
import speech_recognition as sr
from dotenv import load_dotenv
from openai import OpenAI
import os
import asyncio
import edge_tts
from playsound import playsound
from langdetect import detect # Or use lingua-py, fasttext

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

# Load API key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="🎤 Voice Chatbot", layout="centered")
st.title("🎙️ Voice-to-Text + LangChain Streaming (OpenAI)")

# Record voice using speech_recognition and save to file
def record_voice_with_microphone():
    r = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        st.info("Recording... Speak now.")
        audio_data = r.listen(source, phrase_time_limit=20)
        st.success("Recording complete!")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        with open(f.name, "wb") as audio_file:
            audio_file.write(audio_data.get_wav_data())
        return f.name

# Transcribe with Whisper
def transcribe_audio(file_path):
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcript.text

# Stream GPT response
def stream_chat_response(prompt):
    sentence_endings = (".", "!", "?")
    st.subheader("🤖 AI Response")
    response_container = st.empty()
    full_response = ""
    for chunk in client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta.content:
                full_response += delta.content
                print(full_response)
                response_container.markdown(full_response + "▌")
    response_container.markdown(full_response)


def stream_chat_response_and_speak(prompt):
    messages = [{"role": "user", "content": prompt}]
    response_container = st.empty()
    full_response = ""
    chunk_buffer = ""
    sentence_endings = (".", "!", "?")
    for chunk in client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        stream=True):
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            chunk_buffer += token
            response_container.markdown(full_response + "▌")

            # Speak every ~20 chars (you can adjust)
            if any(chunk_buffer.endswith(e) for e in sentence_endings) and len(chunk_buffer.split()) >= 20:
                asyncio.run(speak_edge(chunk_buffer))
                chunk_buffer = ""

    if chunk_buffer:
        asyncio.run(speak_edge(chunk_buffer))  # speak remaining tail

    response_container.markdown(full_response)


# Button action
if st.button("🎤 Record & Ask"):
    path = record_voice_with_microphone()
    st.audio(path)

    st.subheader("📝 Transcription")
    question = transcribe_audio(path)
    st.success(f"User said: **{question}**")

    stream_chat_response_and_speak(question)
