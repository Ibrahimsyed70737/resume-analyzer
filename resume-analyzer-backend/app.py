from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS to handle cross-origin requests
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io # Required for handling file-like objects

app = Flask(__name__)
# Enable CORS for all origins. In a production environment, you should restrict this
# to only the domain where your React frontend is hosted (e.g., origins=["http://localhost:3000"]).
# Enable CORS for the specific origin of your React frontend
# In a production environment, replace 'http://localhost:3000' with your actual frontend domain.
CORS(app, origins=["http://localhost:3000"])

# Function to extract text from PDF
def extract_text_from_pdf(file_buffer):
    """
    Extracts text from a PDF file buffer.

    Args:
        file_buffer: A file-like object (e.g., from request.files) representing the PDF.

    Returns:
        str: The extracted text from the PDF.
    """
    try:
        pdf_reader = PdfReader(file_buffer)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}") # Print to backend console
        return ""

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes_content):
    """
    Ranks resumes based on their similarity to a job description using TF-IDF and Cosine Similarity.

    Args:
        job_description (str): The text of the job description.
        resumes_content (list): A list of strings, where each string is the extracted text of a resume.

    Returns:
        numpy.ndarray: An array of cosine similarity scores for each resume.
    """
    documents = [job_description] + resumes_content
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    vectors = tfidf_matrix.toarray()

    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]

    cosine_similarities = cosine_similarity([job_description_vector], resume_vectors).flatten()
    return cosine_similarities

# API endpoint to handle resume analysis
@app.route('/analyze-resumes', methods=['POST'])
def analyze_resumes():
    if 'job_description' not in request.form:
        return jsonify({"error": "Job description is required."}), 400
    if 'resumes' not in request.files:
        return jsonify({"error": "No resume files uploaded."}), 400

    job_description = request.form['job_description']
    uploaded_files = request.files.getlist('resumes')

    resumes_content = []
    resume_names = []

    for file in uploaded_files:
        if file.filename.endswith('.pdf'):
            # Read the file into an in-memory buffer
            file_buffer = io.BytesIO(file.read())
            text = extract_text_from_pdf(file_buffer)
            if text:
                resumes_content.append(text)
                resume_names.append(file.filename)
            else:
                print(f"Skipping {file.filename} due to text extraction error.")
        else:
            print(f"Skipping {file.filename} due to unsupported file type.")

    if not resumes_content:
        return jsonify({"error": "No valid text could be extracted from any uploaded PDF resumes."}), 400

    scores = rank_resumes(job_description, resumes_content)

    # Prepare results for JSON response
    results = []
    for i, score in enumerate(scores):
        results.append({
            "resume_name": resume_names[i],
            "match_score": float(score * 100) # Convert to percentage and float
        })

    # Sort results by score in descending order
    results.sort(key=lambda x: x['match_score'], reverse=True)

    return jsonify({"results": results}), 200

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)
