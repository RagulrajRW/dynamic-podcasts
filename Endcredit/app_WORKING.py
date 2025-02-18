import os
import glob
import subprocess
import cv2
import pytesseract
import spacy
import csv
from flask import Flask, request, send_file, flash, redirect, url_for, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['FRAME_FOLDER'] = 'frames'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['SECRET_KEY'] = 'your-secret-key'  # Flask session secret key (required for flashing messages)

nlp = spacy.load("en_core_web_trf")

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extract frames from a video at 1-second intervals.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = f'ffmpeg -i "{video_path}" -vf fps={frame_rate} "{output_dir}/frame_%04d.png"'
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"Frames extracted to {output_dir}")
    except subprocess.CalledProcessError as e:
        raise Exception(f"Error extracting frames: {e}")

def extract_text_from_frame(image_path):
    """
    Preprocess an image and extract text using OCR.
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng")
    return text

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

def read_names_from_file(file_path, full_name=False):
    with open(file_path, 'r') as file:
        if full_name:
            names = [line.strip().lower() for line in file.readlines() if line.strip()]
        else:
            names = set(line.strip().lower() for line in file.readlines() if line.strip())
    return names

def compare_names(reference_names, extracted_names):
    """
    Compare reference names with extracted names and return all matching names.
    A match is found if any part (first or last name) of an extracted name exists in the reference names.
    """
    matching_names = []
    for extracted_name in extracted_names:
        extracted_parts = extracted_name.split()  
        for ref_name in reference_names:
            if any(part.lower() == ref_name.lower() for part in extracted_parts):  
                matching_names.append(extracted_name)  
                break  
    return matching_names

def write_common_names_to_csv(common_names, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Common Names"])  
        for name in sorted(common_names):  
            writer.writerow([name])

# Updated route for serving the index.html from the root directory
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle video file upload
        file = request.files.get('video')

        if not file:
            flash("No video file uploaded!", "error")
            return redirect(request.url)
        
        # Save uploaded video
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(video_path)

        try:
            # Extract frames from the video
            extract_frames(video_path, app.config['FRAME_FOLDER'])

            # Extract text from each frame
            extracted_text = []
            for frame in sorted(glob.glob(f"{app.config['FRAME_FOLDER']}/*.png")):
                extracted_text.append(extract_text_from_frame(frame))

            # Combine all extracted text
            combined_text = "\n".join(extracted_text)

            # Extract names from the combined text
            extracted_names = extract_names_from_text(combined_text)

            # Save the extracted names to a text file
            extracted_names_file = os.path.join(app.config['OUTPUT_FOLDER'], 'extracted_names.txt')
            with open(extracted_names_file, 'w') as file:
                file.writelines([name + "\n" for name in extracted_names])

            # Read the reference file (fixed path)
            reference_file_path = "/Users/srragulraj/Desktop/html5up-aerial/frames/processed_names.csv"
            reference_names = read_names_from_file(reference_file_path, full_name=False)

            # Compare extracted names with reference names
            common_names = compare_names(reference_names, [name.lower() for name in extracted_names])

            # Save common names to CSV
            common_names_csv = os.path.join(app.config['OUTPUT_FOLDER'], 'common_names.csv')
            write_common_names_to_csv(common_names, common_names_csv)

            # Cleanup frames
            for frame in glob.glob(f"{app.config['FRAME_FOLDER']}/*.png"):
                os.remove(frame)

            # Return the common names CSV as a downloadable file
            return send_file(common_names_csv, as_attachment=True)

        except Exception as e:
            flash(f"An error occurred: {str(e)}", "error")
            return redirect(request.url)

    # Serve index.html from the root directory
    return send_from_directory('.', 'index.html')

if __name__ == "__main__":
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['FRAME_FOLDER'], exist_ok=True)
    os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
    app.run(debug=True)
