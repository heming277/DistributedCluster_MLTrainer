# app/views.py
from flask import Blueprint, request, jsonify
from .utils import get_job_status, get_job_result
from app.ml import train_pytorch_model, train_tensorflow_model, train_sklearn_model
from app.celery_worker import celery
from celery.result import AsyncResult

main = Blueprint('main', __name__)

@main.route('/')
def hello_world():
    return 'Hello, World!'

@main.route('/submit_job', methods=['POST'])
def submit_job():
    job_data = request.json
    # Determine which training function to call based on the job_data
    if job_data['model_type'] == 'pytorch':
        task = celery.send_task('app.ml.train_pytorch_model', args=[job_data])
    elif job_data['model_type'] == 'tensorflow':
        task = celery.send_task('app.ml.train_tensorflow_model', args=[job_data])
    elif job_data['model_type'] == 'sklearn':
        task = celery.send_task('app.ml.train_sklearn_model', args=[job_data])
    else:
        return jsonify({"error": "Invalid model type specified"}), 400

    return jsonify({"message": "Job submitted successfully", "job_id": task.id}), 202


@main.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    task = AsyncResult(job_id, app=celery)
    response = {
        'job_id': job_id,
        'status': task.status,
        'result': task.result if task.ready() else None
    }
    return jsonify(response), 200

@main.route('/job_result/<job_id>', methods=['GET'])
def job_result(job_id):
    task = AsyncResult(job_id, app=celery)
    if task.ready():
        response = {
            'job_id': job_id,
            'status': task.status,
            'result': task.result
        }
        return jsonify(response), 200
    else:
        response = {
            'job_id': job_id,
            'error': 'Result not found or job still processing',
            'status': task.status
        }
        return jsonify(response), 404