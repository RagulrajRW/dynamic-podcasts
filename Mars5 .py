import os
import google.generativeai as genai
import torch
import torchaudio
from pydub import AudioSegment
import IPython.display as ipd
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)

deep_clone = False
cfg = config_class(deep_clone=deep_clone, top_k=100, temperature=0.7, freq_penalty=3)

output_wav_path = '/Users/srragulraj/Desktop/audio.wav'  
output_mp3_path = '/Users/srragulraj/Desktop/audio.mp3'  

print("Bot: Hello")
print()

def synthesize_audio(response_text):
    max_length = 150 
    response_text = response_text[:max_length]
    
    try:
        dummy_audio = torch.zeros(int(mars5.sr * 0.5))  
        ref_transcript = "" 
        
        print(f"Generating audio for: '{response_text}'")
        _, wav_out = mars5.tts(response_text, dummy_audio, ref_transcript, cfg=cfg)
        
        print("Saving audio as WAV...")
        torchaudio.save(output_wav_path, wav_out.unsqueeze(0), sample_rate=mars5.sr)
        print(f"Synthesized audio saved to {output_wav_path}")
        
        print("Converting WAV to MP3...")
        audio = AudioSegment.from_wav(output_wav_path)
        audio.export(output_mp3_path, format="mp3")
        print(f"Converted audio saved as MP3 to {output_mp3_path}")

    except Exception as e:
        print("An error occurred during audio generation or saving:", str(e))

while True:
    user_input = input("You: ")
    print()

    response = chat_session.send_message(user_input)
    model_response = response.text
    print(f'Bot: {model_response}')
    print()

    chat_session.history.append({"role": "user", "parts": [user_input]})
    chat_session.history.append({"role": "model", "parts": [model_response]})

    synthesize_audio(model_response)
