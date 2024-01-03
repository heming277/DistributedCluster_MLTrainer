from flask import Blueprint, request, jsonify

main = Blueprint('main', __name__)

@main.route('/submit_job', methods=['POST'])
def submit_job():
    job_data = request.json
    # You would typically validate the job data and then queue it for processing
    # For now, let's just return a mock response
    return jsonify({"message": "Job submitted successfully", "job_id": "12345"}), 200
