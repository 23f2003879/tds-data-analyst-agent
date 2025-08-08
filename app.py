from flask import Flask, request, jsonify
from utils import process_question
import time

app = Flask(__name__)

@app.route("/api/", methods=["POST"])
def analyze():
    start_time = time.time()
    
    if 'questions.txt' not in request.files:
        return jsonify({"error": "Missing questions.txt"}), 400

    question_file = request.files['questions.txt'].read().decode("utf-8")
    files = {k: v for k, v in request.files.items() if k != "questions.txt"}

    try:
        result = process_question(question_file, files)
        assert time.time() - start_time < 180
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
