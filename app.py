from flask import Flask, request, jsonify
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

app = Flask(__name__)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route("/")
def home():
    return open("index.html", encoding="utf-8").read()

@app.route("/analyze", methods=["POST"])
def analyze():
    resume_text = request.json["resume"]
    job_role = request.json["role"]

    prompt = f"""
    You are an expert resume reviewer and career coach with 10 years of experience hiring for top tech companies.

    Here are examples of how you analyze resumes:

    EXAMPLE 1:
    Resume: "John, 2nd year student, knows Python basics, made a todo app"
    Role: AI Engineer
    STRENGTHS: Has Python foundation which is required for AI Engineering
    WEAKNESSES: No AI/ML exposure, no real projects, too junior for the role
    IMPROVEMENTS: Learn APIs, build one AI project using OpenAI or Groq, add it to GitHub
    SCORE: 2/10 — Too early stage for AI Engineering role

    EXAMPLE 2:
    Resume: "Sarah, final year, built RAG system, knows LangChain, deployed 2 AI apps, 1 internship at startup"
    Role: AI Engineer  
    STRENGTHS: Hands-on RAG experience, deployed real apps, internship validates skills
    WEAKNESSES: No system design experience mentioned, no large company exposure
    IMPROVEMENTS: Add metrics to projects, learn basic MLOps, contribute to open source
    SCORE: 7/10 — Strong candidate with real experience

    Now analyze this resume using the EXACT same format:

    Candidate Role: {job_role}
    Experience Level: {request.json.get('level', 'fresher')}

    Resume:
    {resume_text}
    Before analyzing, think through these questions internally:
    1. What level is this candidate really at?
    2. Does their experience match the role they're applying for?
    3. What's the single biggest gap between where they are and where they need to be?
    4. What 3 specific actions would move them closest to the role?

    Then respond in the format below.

    Respond in EXACTLY this format:
    STRENGTHS:
    [your analysis]

    WEAKNESSES:
    [your analysis]

    IMPROVEMENTS:
    [your analysis]

    SCORE: X/10
    [reason]
    """

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    return jsonify({"analysis": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)