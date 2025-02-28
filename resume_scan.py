print("Loading....")

import os
import PyPDF2
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('punkt')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()

# Function to preprocess text
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())  # Convert to lowercase & tokenize
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(filtered_words)

# Load resume (PDF file)
resume_path = r"C:\Users\LENOVO\OneDrive\Desktop\Temp\Shivani Resume.pdf"

if not os.path.exists(resume_path):
    print("Error: Resume file not found!")
    exit()

resume_text = extract_text_from_pdf(resume_path)
resume_text = preprocess_text(resume_text)

# Load job description (TXT file)
job_desc_path = r"C:\Users\LENOVO\OneDrive\Desktop\Temp\job_description.txt"

if not os.path.exists(job_desc_path):
    print("Error: Job description file not found!")
    exit()

with open(job_desc_path, "r", encoding="utf-8") as file:
    job_description = preprocess_text(file.read())


# Compare similarity using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([job_description, resume_text])
similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

# Print similarity result
print(f"Resume Similarity Score with Job Description: {similarity_score:.5f}")

if similarity_score > 0.75:
    print("✅  Your resume is a strong match for the job!")
elif similarity_score > 0.50:
    print("⚠️  Your resume is a decent match. Consider improving it.")
elif similarity_score > 0.30:
    print("⚠️ ⚠️  Your resume is a decent match. Consider improving it more.")
else:
    print("❌  Your resume is not a good match. Try adding relevant skills and experience.")