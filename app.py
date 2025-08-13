from flask import Flask, request, jsonify
from utils import process_question
import time

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return "✅ Data Analyst Agent is running", 200

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in JSON body"}), 400

    question = data["question"]
    try:
        answer = process_question(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/", methods=["POST"])
def analyze():
    start_time = time.time()

    question_file = None
    for key in request.files:
        if key.lower() == "questions.txt":
            question_file = request.files[key].read().decode("utf-8")
            break

    if not question_file:
        return jsonify({"error": "Missing questions.txt"}), 400

    files = {k: v for k, v in request.files.items() if k.lower() != "questions.txt"}

    try:
        result = process_question(question_file, files)
        if time.time() - start_time > 180:
            return jsonify({"error": "Processing exceeded 3-minute limit"}), 500
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
