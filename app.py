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
You are an expert resume reviewer.
Analyze this resume for a {job_role} position at {request.json.get('level', 'fresher')} level.

Respond in EXACTLY this format:
STRENGTHS:
[list strengths here]

WEAKNESSES:
[list weaknesses here]

IMPROVEMENTS:
[list specific improvements here]

SCORE: X/10
[reason for score]

Resume:
{resume_text}
"""

    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )

    return jsonify({"analysis": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)