from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import os

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('questions.csv')

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Topic'])

def get_questions_and_answers(topic, num_questions):
    topic_vector = vectorizer.transform([topic])
    similarities = cosine_similarity(topic_vector, X).flatten()
    similar_indices = similarities.argsort()[-num_questions:][::-1]
    
    questions = df['Question'].iloc[similar_indices].tolist()
    answers = df['Answer'].iloc[similar_indices].tolist()  
    
    return list(zip(questions, answers)) if questions else [("No related questions found.", "No answers available.")]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_question', methods=['POST'])
def generate_question():
    data = request.get_json()
    topic = data.get('topic')
    num_questions = int(data.get('num_questions', 5))

    if num_questions <= 0:
        return jsonify({"error": "Number of questions must be a positive integer."}), 400

    question_answer_pairs = get_questions_and_answers(topic, num_questions)
    return jsonify({"qa_pairs": question_answer_pairs})

def generate_pdf(qa_pairs, include_answers, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "Generated Question Paper", ln=True, align="C")
    pdf.set_font("Arial", "", 12)

    for i, (question, answer) in enumerate(qa_pairs, 1):
        pdf.cell(0, 10, f"{i}. {question}", ln=True)
        if include_answers:
            pdf.multi_cell(0, 10, f"Answer: {answer}\n", align="L")

    pdf.output(filename)
    return filename

@app.route('/download_questions_pdf', methods=['POST'])
def download_questions_pdf():
    data = request.get_json()
    topic = data.get('topic')
    num_questions = int(data.get('num_questions', 5))

    if num_questions <= 0:
        return jsonify({"error": "Number of questions must be a positive integer."}), 400

    question_answer_pairs = get_questions_and_answers(topic, num_questions)
    filename = f"{topic}_questions.pdf"
    generate_pdf(question_answer_pairs, include_answers=False, filename=filename)
    return send_file(filename, as_attachment=True)

@app.route('/download_questions_answers_pdf', methods=['POST'])
def download_questions_answers_pdf():
    data = request.get_json()
    topic = data.get('topic')
    num_questions = int(data.get('num_questions', 5))

    if num_questions <= 0:
        return jsonify({"error": "Number of questions must be a positive integer."}), 400

    question_answer_pairs = get_questions_and_answers(topic, num_questions)
    filename = f"{topic}_questions_with_answers.pdf"
    generate_pdf(question_answer_pairs, include_answers=True, filename=filename)
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)



