import flask
from flask_cors import CORS
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from teacher_allocation import process_input  # Your Python allocation function

load_dotenv()

app = Flask(__name__)
CORS(app)

@app.route('/get-opencage-key', methods=['GET'])
def get_opencage_key():
    api_key = os.getenv("OPENCAGE_API_KEY")
    if not api_key:
        return jsonify({'error': 'API key not found'}), 500
    return jsonify({'key': api_key})

@app.route("/allocate", methods=["POST"])
def allocate():
    data = request.json
    schools = data.get("schools", [])

    # Call your Python function to process the allocation
    suggestions = process_input(schools)

    return jsonify(suggestions)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
