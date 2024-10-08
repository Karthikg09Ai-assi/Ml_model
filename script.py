import warnings
import json
from transformers import pipeline
from PyPDF2 import PdfReader

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Step 1: Load the JSON file and extract the document path
with open("config.json", "r") as file:
    config = json.load(file)

pdf_path = config["document_path"]

# Step 2: Extract text from PDF
reader = PdfReader(pdf_path)
text = ""

for page in reader.pages:
    text += page.extract_text()

# Step 3: Initialize the text classification pipeline with DistilBERT
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Step 4: Define a function to split text into chunks
def split_text(text, chunk_size=512):
    """Splits the input text into chunks of specified token size."""
    words = text.split()  # Split text into words
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

# Step 5: Split text and classify each chunk
chunks = split_text(text)
classified_sections = []

for chunk in chunks:
    # Use the classifier with truncation and specify max_length
    result = classifier(chunk, truncation=True, max_length=512)
    classified_sections.append((result[0]['label'], chunk))

# Step 6: Export identified sections with UTF-8 encoding
output_file = r"D:\gayathri\USB PD R3.2 V1.0\extracted_sections.txt"
with open(output_file, "w", encoding="utf-8") as file:
    for label, section in classified_sections:
        file.write(f"{label}: {section}\n\n")

