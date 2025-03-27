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

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = "sqlite:///history.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.secret_key = ""

# Load the dataset
df = pd.read_csv('questions.csv')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["Topic"])

# Configure Google Gemini API
genai.configure(api_key=app.secret_key)

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

def get_questions_and_answers(topic, num_questions):
    # Generate similarity scores for the topic
    topic_vector = vectorizer.transform([topic])
    similarities = cosine_similarity(topic_vector, X).flatten()

    # Get indices sorted by similarity (highest first)
    sorted_indices = similarities.argsort()[::-1]

    # Filter relevant questions based on a similarity threshold
    relevant_questions = df.iloc[sorted_indices]
    relevant_questions = relevant_questions[relevant_questions["Topic"].str.lower() == topic.lower()]

    # Ensure unique questions
    unique_questions = relevant_questions["Question"].drop_duplicates()

    # Check if there are enough questions
    if len(unique_questions) < num_questions:
        num_questions = len(unique_questions)

    # Randomly sample the required number of questions
    sampled_questions = unique_questions.sample(n=num_questions)

    return sampled_questions.tolist()
  

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
    dataset_qa = get_questions_and_answers(topic, num_questions // 2)

    # Generate Gemini questions
    gemini_qa = generate_gemini_questions(topic, num_questions - len(dataset_qa))

    # Combine dataset and Gemini questions
    questions = dataset_qa + gemini_qa

    # Store the questions in the session
    session['questions'] = questions

    return questions


@app.route("/generate_question", methods=["POST"])
def generate_question():
    data = request.get_json()
    topic = data.get("topic")
    num_questions = max(1, int(data.get("num_questions", 5)))

    # Generate and store combined questions in session
    questions = generate_combined_questions(topic, num_questions)

    print(f"Combined Questions: {questions}")  # Debugging

    return jsonify({"questions": questions})



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
    
