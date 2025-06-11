from flask import Flask, request, jsonify, session, render_template, send_file, redirect, url_for
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import google.generativeai as genai
import os
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import smtplib
from itsdangerous import URLSafeTimedSerializer
from email.mime.text import MIMEText
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import Dataset
import markdown
from sklearn.linear_model import LogisticRegression
import joblib  # For saving the model

# Load dataset
df = pd.read_csv("questions.csv")
df["Category"] = df["Topic"]  # Use topic as category label for now
df.to_csv("questions_labeled.csv", index=False)

# Feature & label
X = df['Topic'] + " " + df['Question']
y = df['Category']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Model
model_ = LogisticRegression()
model_.fit(X_train, y_train)

# Save model and vectorizer
joblib.dump(model_, 'model_.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///history.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.secret_key = " "


try: 
# Configure Google Gemini API
    genai.configure(api_key=app.secret_key)
# These settings are aggressive to block harmful content. Adjust as needed.
    generation_config = {
        "temperature": 0.7,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    print("Gemini Pro Model configured successfully.")

except KeyError:
    print("🔴 ERROR: GOOGLE_API_KEY not found in environment variables.")
    print("Please create a .env file and add your key.")
    model = None
except Exception as e:
    print(f"🔴 ERROR: An unexpected error occurred during API configuration: {e}")
    model = None
    
# Load the dataset
# df = pd.read_csv('questions.csv')

# # Initialize TF-IDF Vectorizer
# vectorizer = TfidfVectorizer(stop_words="english")
# X = vectorizer.fit_transform(df["Topic"])

class History(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    topic = db.Column(db.String(200), nullable=False)
    date_searched = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.topic}"

# Create the database tables if they don't exist
with app.app_context():
    db.create_all()

import random

# def get_questions_and_answers(topic, num_questions):
#     # Generate similarity scores for the topic
#     topic_vector = vectorizer.transform([topic])
#     similarities = cosine_similarity(topic_vector, X).flatten()

#     # Get indices sorted by similarity (highest first)
#     sorted_indices = similarities.argsort()[::-1]

#     # Filter relevant questions based on a similarity threshold
#     relevant_questions = df.iloc[sorted_indices]
#     relevant_questions = relevant_questions[relevant_questions["Topic"].str.lower() == topic.lower()]

#     # Ensure unique questions
#     unique_questions = relevant_questions["Question"].drop_duplicates()

#     # Check if there are enough questions
#     if len(unique_questions) < num_questions:
#         num_questions = len(unique_questions)

#     # Randomly sample the required number of questions
#     sampled_questions = unique_questions.sample(n=num_questions)

#     return sampled_questions.tolist()

def remove_similar_questions(questions, threshold=0.85):
    vectors = vectorizer.transform(questions)
    unique = []
    for i, q in enumerate(questions):
        is_similar = False
        for uq in unique:
            sim = cosine_similarity(vectorizer.transform([q]), vectorizer.transform([uq]))[0][0]
            if sim > threshold:
                is_similar = True
                break
        if not is_similar:
            unique.append(q)
    return unique


def get_ml_questions(topic, num_questions):
    input_vec = vectorizer.transform([topic])
    predicted_category = model_.predict(input_vec)[0]

    # Filter questions from that category
    matching_questions = df[df['Category'] == predicted_category]['Question'].sample(n=num_questions, replace=True).tolist()
    return matching_questions
  

def generate_gemini_questions(topic, num_questions):
    if num_questions <= 0:
        return []
    try:
        prompt = f"Generate {num_questions} unique and well-structured questions about {topic}. Each question should end with a \n for example questions would be in format: what is orchestration? \n define cloud. \n ....and so on. This would be the format. No special symbol will be used unnecessary except . , ? and question numbering shouldn't be there just seperate each question with \n."
        response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
        questions = response.text.split("\n")  # Split response into separate questions
        questions = [q.strip() for q in questions if q.strip()]  # Clean up empty strings
        return questions[:num_questions]  # Limit to requested number
    except Exception as e:
        print(f"Error generating Gemini questions: {e}")
        return []
    
def generate_combined_questions(topic, num_questions):
    # Generate dataset-based questions
    dataset_qa = get_ml_questions(topic, num_questions // 2)

    # Generate Gemini questions
    gemini_qa = generate_gemini_questions(topic, num_questions - len(dataset_qa))

    # Combine dataset and Gemini questions
    questions = dataset_qa + gemini_qa

    # Store the questions in the session
    session['questions'] = questions

    return questions

def get_unique_questions(topic, num_questions):
    max_attempts = 5
    questions = []

    # Initial fill from dataset and Gemini
    dataset_qs = get_ml_questions(topic, num_questions // 2)
    gemini_qs = generate_gemini_questions(topic, num_questions - len(dataset_qs))
    combined = dataset_qs + gemini_qs
    
    print(f"First Combined Questions: {combined}")

    # Remove similar questions
    questions = remove_similar_questions(combined)

    attempt = 1
    while len(questions) < num_questions and attempt <= max_attempts:
        needed = num_questions - len(questions)
        print(f"[Attempt {attempt}] Need {needed} more unique questions")
        more_qs = generate_gemini_questions(topic, needed * 2)  # ask more to account for overlap
        temp_combined = questions + more_qs
        questions = remove_similar_questions(temp_combined)
        attempt += 1

    return questions[:num_questions]  # Trim if extra



@app.route("/generate_question", methods=["POST"])
def generate_question():
    data = request.get_json()
    topic = data.get("topic")
    num_questions = max(1, int(data.get("num_questions", 5)))

    questions = get_unique_questions(topic, num_questions)
    print(f"Final Combined Questions: {questions}")
    session['questions'] = questions
    session['topic'] = topic

    return jsonify({"questions": questions})
    
    
    
    # data = request.get_json()
    # topic = data.get("topic")
    # num_questions = max(1, int(data.get("num_questions", 5)))

    # # Generate and store combined questions in session
    # questions = generate_combined_questions(topic, num_questions)

    # print(f"Combined Questions: {questions}")  # Debugging

    # return jsonify({"questions": questions})



def generate_pdf(questions, filename, topic):
    pdf = FPDF()
    heading = f"{topic} Question Paper.pdf"
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, heading, ln=True, align="C")
    pdf.set_font("Arial", "", 12)

    for i, question in enumerate(questions, 1):
        pdf.cell(10, 10, f"{i}. ", ln=False)
        pdf.multi_cell(0, 10, question) 

    pdf.output(filename)
    return filename

@app.route("/download_questions_pdf", methods=["POST"])
def download_questions_pdf():
    data = request.get_json()
    topic = data.get("topic")
    num_questions = max(1, int(data.get("num_questions", 5)))

    # Retrieve the questions from the session
    questions = session.get('questions', None)

    if not questions:
        # If questions are not in session, regenerate them
        questions = generate_combined_questions(topic, num_questions)

    print(f"Combined Questions for PDF: {questions}")  # Debugging

    # Generate and return PDF
    filename = f"{topic}_questions.pdf"
    generate_pdf(questions, filename, topic)
    return send_file(filename, as_attachment=True)


@app.route("/generate_answers", methods=["POST"])
def generate_answers():
    data = request.get_json()
    questions = data.get("questions", [])

    answers = []
    for question in questions:
        try:
            prompt = f"Answer this question briefly and clearly:\nQ: {question}"
            response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        answers.append({"question": question, "answer": answer})

    return jsonify({"qa_pairs": answers})


@app.route('/Customize Question Paper')
def customize():
    return render_template('customize.html')


# Function to create the prompt
def generate_prompt(subject, chapter, num_q, difficulty, qtype):
    return (
        f"Generate {num_q} {difficulty}-level {qtype} questions "
        f"for the subject '{subject}', chapter '{chapter}'.\n\n"
        "Number the questions clearly."
    )
    
    
# Generate questions route
@app.route("/generate", methods=["POST"])
def generate():
    subject    = request.form["subject"]
    chapter    = request.form["chapter"]
    num_q      = request.form["num_questions"]
    difficulty = request.form["difficulty"]
    qtype      = request.form["qtype"]

    prompt = generate_prompt(subject, chapter, num_q, difficulty, qtype)

    try:
        # Generate content using Gemini
        response = model.generate_content(prompt)
        questions_md = response.text

        # Convert Markdown to HTML
        questions_html = markdown.markdown(
            questions_md,
            extensions=["fenced_code", "nl2br"]
        )
    except Exception as e:
        questions_html = f"<p><strong>Error calling Gemini API:</strong> {e}</p>"

    return render_template("customize.html", questions=questions_html)


@app.route('/ask', methods=['POST'])
def ask():
    """Handles the question from the user and returns the model's answer."""
    if not model:
        return jsonify({'error': 'The Gemini API is not configured. Please check the server logs.'}), 500

    data = request.get_json()
    user_question = data.get('question')

    if not user_question:
        return jsonify({'error': 'Question cannot be empty.'}), 400

    try:
        # --- Prompt Engineering ---
        # We wrap the user's question with a persona and instructions for better results.
        prompt = f"""
        As an expert academic tutor, your role is to provide a clear, accurate, and helpful answer to the following student's question.
        Break down complex topics into easy-to-understand parts. Use lists, bold text, or examples where it helps with clarity.
        
        Student's Question: "{user_question}"
        
        Your Answer:
        """
        
        # Generate the response
        response = model.generate_content(prompt)
        
        # Check for safety blocks or empty responses
        if not response.parts:
            # This can happen if the prompt is blocked by safety filters.
            answer_text = "I'm sorry, I couldn't generate a response for that question. It might have violated the safety guidelines. Please try asking a different question."
        else:
            # Convert Markdown response to HTML
            answer_text = markdown.markdown(response.text, extras=["fenced-code-blocks", "tables"])

        return jsonify({'answer': answer_text})

    except Exception as e:
        # Generic error handler for other API issues
        print(f"🔴 ERROR during Gemini API call: {e}")
        return jsonify({'error': f'An unexpected error occurred while processing your request: {e}'}), 500


@app.route('/', methods=['GET', 'POST'])
def history_():
    if request.method == 'POST':
        topic = request.form['topic']
        print(f"Received topic: {topic}")  # Debugging
        if topic:  # Ensure topic is not empty
            new_history = History(topic=topic)
            db.session.add(new_history)
            try:
                db.session.commit()
                print(f"Added to history: {new_history}")  # Debugging
            except Exception as e:
                print(f"Error adding to history: {e}")  # Debugging
                db.session.rollback()  # Rollback in case of error

    all_history = History.query.all() 
    return render_template('index.html', allHistory=all_history)

@app.route('/show')
def show():
    all_history = History.query.all()
    print(all_history)  # Debugging
    return render_template('history.html', allHistory=all_history)

@app.route('/delete/<int:sno>')
def delete(sno):
    history_entry = History.query.filter_by(sno=sno).first()
    if history_entry:
        db.session.delete(history_entry)
        db.session.commit()
        print(f"Deleted history entry: {history_entry}")  # Debugging
    return redirect("/")

@app.route('/Home')
def home():
    return render_template('index.html')

@app.route('/Contact Us')
def contact():
    return render_template('Contact.html')


# Outlook SMTP Configuration
smtp_server = 'smtp.gmail.com'
smtp_port = 587  # TLS port
sender_email = 'mtesting12012025@gmail.com'  # Replace with your Outlook email
sender_password = 'wavl clto uznq gigx'  # Replace with your password or app password

# Secret Key for URL Encoding
app.secret_key_ = 'Ajhfhhgs9772hhdhajaj'  # Replace with your secret key

# Initialize Serializer for Token Generation
s = URLSafeTimedSerializer(app.secret_key_)

# Generate Verification Code (Token)
def generate_verification_token(email):
    return s.dumps(email, salt='email-confirm')

# Send the verification email
def send_verification_email(email, token):
    verification_link = url_for('confirm_email', token=token, _external=True)
    subject = "Please confirm your email"
    body = f"Click the following link to confirm your email: {verification_link}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = email

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)  # Login to your Outlook account
            server.sendmail(sender_email, email, msg.as_string())  # Send the email
        print("Verification email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")
        
@app.route('/Register-Login', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        # Render the registration form
        return render_template('register-login.html')

    elif request.method == 'POST':
        # Handle the form submission
        email = request.form['email']
        print(f"Received email: {email}")  # Debugging
        token = generate_verification_token(email)
        print(f"Generated token: {token}")  # Debugging
        send_verification_email(email, token)
    return redirect(url_for('verify_email'))
@app.route('/verify-email')
def verify_email():
    return render_template('verify-email.html')  # A simple page to ask for the code

@app.route('/confirm_email/<token>')
def confirm_email(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=3600)  # Token expires after 1 hour
        return f"Email {email} confirmed successfully!"
    except Exception as e:
        return "The token is invalid or has expired."

if __name__ == "__main__":
    app.run(debug=True)
    
