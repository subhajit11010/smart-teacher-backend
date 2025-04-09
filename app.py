import flask
from flask_cors import CORS
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from teacher_allocation import process_input  # Your Python allocation function

load_dotenv()

app = Flask(__name__)

# ✅ CORS config: allow only your frontend domain
CORS(app, resources={r"/*": {"origins": "https://smart-teacher-reallocation.vercel.app"}})

@app.route('/get-opencage-key', methods=['GET'])
def get_opencage_key():
    api_key = os.getenv("OPENCAGE_API_KEY")
    if not api_key:
        return jsonify({'error': 'API key not found'}), 500
    return jsonify({'key': api_key})

@app.route("/allocate", methods=["POST"])
def allocate():
    try:
        data = request.get_json()
        schools = data.get("schools", [])
        
        # Optional: Log for debugging
        print("Received schools data:", schools)

        # Call your Python function to process the allocation
        suggestions = process_input(schools)

        return jsonify(suggestions)
    
    except Exception as e:
        print("Error during allocation:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
