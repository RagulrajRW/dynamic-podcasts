import os
import numpy as np
import subprocess
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.hparams import sampling_rate
from sklearn.cluster import AgglomerativeClustering
import whisper
from datetime import timedelta
import yt_dlp 

def download_audio(url, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path, 
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    print(f"Audio downloaded to {output_path}")

def convert_to_wav(input_audio_path, output_wav_path):
    command = ['ffmpeg', '-i', input_audio_path, '-ar', '16000', output_wav_path]
    subprocess.run(command, check=True)
    print(f"Audio converted to WAV format at {output_wav_path}")

def extract_speaker_embeddings(audio_path, encoder):
    wav = preprocess_wav(audio_path)
    embeddings = encoder.embed_utterance(wav)
    return embeddings

def cluster_speakers(embeddings, num_speakers=2):
    embeddings = embeddings.reshape(-1, 1) 
    clustering = AgglomerativeClustering(n_clusters=num_speakers, affinity='euclidean', linkage='ward')
    labels = clustering.fit_predict(embeddings)
    return labels

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["segments"]

def merge_diarization_and_transcription(segments, speaker_labels):
    dialogue = []
    for i, segment in enumerate(segments):
        start_time = str(timedelta(seconds=int(segment["start"])))
        end_time = str(timedelta(seconds=int(segment["end"])))
        speaker = f"Speaker {speaker_labels[i % len(speaker_labels)]}"  
        text = segment["text"]
        dialogue.append(f"{start_time} - {end_time} | {speaker}: {text}")
    return "\n".join(dialogue)

def save_transcript_to_file(transcript, file_path):
    with open(file_path, "w") as file:
        file.write(transcript)
    print(f"Transcript saved to {file_path}")

def main():
    url = input("Enter the video/audio URL: ")
    output_directory = '/Users/srragulraj/Desktop/video podcast' 
    downloaded_audio_path = os.path.join(output_directory, "audio.mp4")  
    output_wav_path = os.path.join(output_directory, "audio.wav")
    transcript_file_path = os.path.join(output_directory, "transcript.txt")

    download_audio(url, downloaded_audio_path)

    convert_to_wav(downloaded_audio_path, output_wav_path)

    encoder = VoiceEncoder()
    embeddings = extract_speaker_embeddings(output_wav_path, encoder)

    speaker_labels = cluster_speakers(embeddings)

    transcription_segments = transcribe_audio(output_wav_path)

    transcript = merge_diarization_and_transcription(transcription_segments, speaker_labels)

    save_transcript_to_file(transcript, transcript_file_path)

if __name__ == "__main__":
    main()
