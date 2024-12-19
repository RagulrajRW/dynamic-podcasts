import os
import glob
import subprocess
import cv2
import pytesseract
import spacy
import json

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at 1-second intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = f'ffmpeg -i "{video_path}" -vf fps={frame_rate} "{output_dir}/frame_%04d.png"'
    subprocess.run(cmd, shell=True, check=True)
    print(f"Frames extracted to {output_dir}")

def extract_text_from_frame(image_path):
    """
    Preprocess an image and extract text using OCR.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng")
    return text

nlp = spacy.load("en_core_web_trf")

def extract_names_from_text(text):
    """
    Identify names from text using spaCy's NER model.
    """
    doc = nlp(text)
    names = []
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            tokens = ent.text.split()
            if len(tokens) > 1: 
                names.append(ent.text)
    return names

def main():
    VIDEO_PATH = "/Users/srragulraj/Desktop/end/inside out ending scene closing credits.mp4"  # Replace with your video path
    OUTPUT_DIR = "frames"
    OUTPUT_TEXT_FILE = "extracted_names.txt"
    OUTPUT_JSON_FILE = "extracted_names.json"
    
    print("Extracting frames...")
    extract_frames(VIDEO_PATH, OUTPUT_DIR)
    
    print("Extracting text from frames...")
    extracted_text = []
    for frame_path in sorted(glob.glob(f"{OUTPUT_DIR}/*.png")):
        text = extract_text_from_frame(frame_path)
        extracted_text.append(text)
    
    combined_text = "\n".join(extracted_text)
    
    print("Identifying names...")
    names = extract_names_from_text(combined_text)
    
    print(f"Saving results to {OUTPUT_TEXT_FILE} and {OUTPUT_JSON_FILE}...")
    with open(OUTPUT_TEXT_FILE, "w") as f:
        for name in names:
            f.write(f"{name}\n")
    with open(OUTPUT_JSON_FILE, "w") as f:
        json.dump({"names": names}, f, indent=4)
    
    print(f"Extracted names saved successfully.\nText File: {OUTPUT_TEXT_FILE}\nJSON File: {OUTPUT_JSON_FILE}")
    
    for file in glob.glob(f"{OUTPUT_DIR}/*.png"):
        os.remove(file)
    os.rmdir(OUTPUT_DIR)

if __name__ == "__main__":
    main()
