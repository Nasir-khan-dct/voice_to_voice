import whisper
import os
from groq import Groq
from gtts import gTTS
import streamlit as st

# Load the Whisper model
model = whisper.load_model("base")

# Function to transcribe audio using Whisper
def transcribe_audio(audio_file):
    result = model.transcribe(audio_file)
    return result['text']

# Initialize Groq API client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Function to get response from Groq's LLM
def get_llm_response(user_input):
    chat_completion = client.chat.completions.create(
        messages=[{
            "role": "user",
            "content": user_input,
        }],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Function to convert text to speech using gTTS
def text_to_speech(text, output_audio_path="response.mp3"):
    tts = gTTS(text)
    tts.save(output_audio_path)
    return output_audio_path

# Streamlit app interface
st.title("Real-Time Voice-to-Voice Chatbot")
st.write("This chatbot transcribes your voice, interacts with an LLM, and responds with audio! Record or upload an audio file.")

# Upload audio file or record via microphone
audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])
if audio_file:
    # Step 1: Transcribe input audio to text
    st.write("Transcribing audio...")
    transcription = transcribe_audio(audio_file)
    st.write(f"Transcription: {transcription}")
    
    # Step 2: Get response from LLM via Groq
    st.write("Generating response...")
    response = get_llm_response(transcription)
    st.write(f"LLM Response: {response}")
    
    # Step 3: Convert the response text to audio
    st.write("Converting response to audio...")
    audio_response_path = text_to_speech(response)
    
    # Play the generated response audio
    audio_file = open(audio_response_path, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/mp3')
