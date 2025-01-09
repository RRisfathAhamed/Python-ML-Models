from flask import Flask, request, jsonify
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import os
import uuid
from tika import parser
from werkzeug.utils import secure_filename
import spacy
from spacy.matcher import Matcher
import re
import os
from groq import Groq
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

# Load the trained model from the pickle file
with open('model/svc_model.pkl', 'rb') as model_file:
    svc_model = pickle.load(model_file)

# Load the TF-IDF vectorizer
with open('model/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

# Load the LabelEncoder (make sure you saved the LabelEncoder during training)
with open('model/label_encoder.pkl', 'rb') as le_file:  # Add LabelEncoder file loading
    le = pickle.load(le_file)

app = Flask(__name__)
# Allow all origins
CORS(app) 


# Define the prediction function
def pred(resume_text):
    # Transform the resume text using the saved TF-IDF vectorizer
    resume_tfidf = tfidf.transform([resume_text])
    # Convert sparse matrix to dense
    resume_tfidf_dense = resume_tfidf.toarray()
    # Predict the category using the trained model
    predicted_category = svc_model.predict(resume_tfidf_dense)
    
    # Convert the predicted category to the corresponding job category name
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the resume text from the POST request
    resume_text = data.get('resumeText', '')
    if not resume_text:
        return jsonify({'error': 'Resume text is required'}), 400

    # Predict the category
    predicted_category = pred(resume_text)
    print(predicted_category)

    return jsonify({'predicted_job': predicted_category})


############################################################################################################

# Load spaCy model
nlp = spacy.load('en_core_web_sm')
matcher = Matcher(nlp.vocab)

# Set the folder where uploaded files will be stored 
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

# Configure the upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    file_data = parser.from_file(file_path)
    return file_data['content']

def get_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.findall(string)

def get_phone_numbers(string):
    r = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
    phone_numbers = r.findall(string)
    return [re.sub(r'\D', '', num) for num in phone_numbers]

def extract_name(text):
    # Get the first line and clean it
    first_lines = [line.strip() for line in text.split('\n') if line.strip()][:2]
    if first_lines:
        return first_lines[0]
    return None

def extract_linkedin(text):
    linkedin_pattern = r'(?:linkedin\.com/in/[\w-]+)'
    linkedin = re.search(linkedin_pattern, text)
    if linkedin:
        return 'www.' + linkedin.group(0)
    return None

def extract_address(text):
    address_pattern = r'\d+\s+[\w\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)[,\s]+[\w\s]+(?:,\s*[\w\s]+)?'
    address = re.search(address_pattern, text)
    if address:
        return address.group(0)
    return None

def split_into_sections(text):
    # Define section headers
    section_headers = {
        'summary': r'(?i)^\s*(?:summary|profile|about)\s*:?\s*$',
        'skills': r'(?i)^\s*(?:skills|expertise|competencies)\s*:?\s*$',
        'experience': r'(?i)^\s*(?:experience|work experience|employment)\s*:?\s*$',
        'education': r'(?i)^\s*education\s*:?\s*$',
        'interests': r'(?i)^\s*(?:interests|hobbies)\s*:?\s*$'
    }
    
    sections = {'header': []}
    current_section = 'header'
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line is a section header
        is_header = False
        for section, pattern in section_headers.items():
            if re.match(pattern, line):
                current_section = section
                sections[current_section] = []
                is_header = True
                break
                
        if not is_header:
            if current_section not in sections:
                sections[current_section] = []
            sections[current_section].append(line)
            
    return sections

def parse_skills(skills_lines):
    skills = []
    for line in skills_lines:
        # Split on common delimiters and capital letters
        candidates = re.split(r'[,•\-\s]+(?=[A-Z])|[,•\-\s]+', line)
        current_skill = ''
        for skill in candidates:
            skill = skill.strip()
            if skill and len(skill) > 2:  # Avoid empty or very short strings
                if current_skill:
                    current_skill += ' '
                current_skill += skill
        if current_skill:
            skills.append(current_skill)
    return skills


def parse_experience(exp_lines):
    experiences = []
    current_exp = {}
    
    for line in exp_lines:
        # Check if line starts with a date pattern (likely a new position)
        date_pattern = r'^(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}'
        if re.match(date_pattern, line):
            if current_exp:
                experiences.append(current_exp)
                current_exp = {}
            current_exp['dates'] = line
        elif not current_exp:
            current_exp = {'title': line}
        elif 'company' not in current_exp:
            current_exp['company'] = line
        else:
            if 'responsibilities' not in current_exp:
                current_exp['responsibilities'] = []
            current_exp['responsibilities'].append(line)
    
    if current_exp:
        experiences.append(current_exp)
    
    return experiences

def parse_education(edu_lines):
    education = []
    current_edu = {}
    
    for line in edu_lines:
        if 'Degree' in line:
            if current_edu:
                education.append(current_edu)
            current_edu = {'degree': line}
        elif 'University' in line or 'College' in line:
            current_edu['institution'] = line
        elif line:
            if 'details' not in current_edu:
                current_edu['details'] = []
            current_edu['details'].append(line)
    
    if current_edu:
        education.append(current_edu)
    
    return education

def parse_cv(text):
    parsed_content = {
        'contact_info': {},
        'sections': {}
    }
    
    # Extract contact information
    parsed_content['contact_info']['name'] = extract_name(text)
    emails = get_email_addresses(text)
    if emails:
        parsed_content['contact_info']['email'] = emails[0]
    phones = get_phone_numbers(text)
    if phones:
        parsed_content['contact_info']['phone'] = phones[0]
    parsed_content['contact_info']['linkedin'] = extract_linkedin(text)
    parsed_content['contact_info']['address'] = extract_address(text)
    
    # Split text into sections
    sections = split_into_sections(text)
    
    # Process each section
    if 'summary' in sections:
        parsed_content['sections']['summary'] = ' '.join(sections['summary'])
    
    if 'skills' in sections:
        parsed_content['sections']['skills'] = parse_skills(sections['skills'])
    
    if 'experience' in sections:
        parsed_content['sections']['experience'] = parse_experience(sections['experience'])
    
    if 'education' in sections:
        parsed_content['sections']['education'] = parse_education(sections['education'])
    
    if 'interests' in sections:
        parsed_content['sections']['interests'] = ' '.join(sections['interests'])
    
    return parsed_content

@app.route('/upload-cv', methods=['POST'])
def upload_cv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only PDF files are allowed'}), 400

    try:
        # Save the uploaded PDF file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(file_path)

        # Extract text from the PDF
        text = extract_text_from_pdf(file_path)

        # Parse the CV content
        parsed_content = parse_cv(text)

        # Clean up: remove the file after processing
        os.remove(file_path)

        return jsonify(parsed_content), 200

    except Exception as e:
        # Clean up file if it exists
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        return jsonify({'error': str(e)}), 500
##################################################################################

# Retrieve the API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

# Function to send a chat request to the Groq API
def chat_with_groq(messages, model="llama-3.3-70b-versatile", temperature=1, max_tokens=1024, top_p=1):
    client = Groq(api_key=GROQ_API_KEY)
    
    # Prepare chat completion request
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stream=True,
        stop=None
    )

    # Collect the response and stream it as chunks
    result = ""
    for chunk in completion:
        result += chunk.choices[0].delta.content or ""
    
    return result

# Define a route for the chat API
@app.route("/chat", methods=["POST"])
def chat():
    try:
        # Get user input from the request
        data = request.get_json()

        # Validate input
        if "messages" not in data:
            return jsonify({"error": "Missing 'messages' field in request body"}), 400

        messages = data["messages"]

        # Call the Groq chat API
        response = chat_with_groq(messages)

        # Return the response from the Groq API
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5001)
