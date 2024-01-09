# app/views.py
from flask import Blueprint, request, jsonify, current_app
from flask import render_template
from flask import send_from_directory
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
from .utils import get_job_status, get_job_result
from app.ml import train_pytorch_model, train_tensorflow_model, train_sklearn_model
from app.celery_worker import celery
from celery.result import AsyncResult
import re

main = Blueprint('main', __name__)

@main.route('/')
def hello_world():
    #frontend_dir = os.path.abspath(os.path.join(current_app.root_path, '..', 'frontend'))
    return render_template('index.html') 

def is_identifier_column(col_name):
    # Normalize the column name by removing whitespace and special characters
    normalized_col_name = re.sub(r'\W+', '', col_name).lower()
    # Check if the normalized column name matches common identifier patterns
    return normalized_col_name in ['id', 'index']

@main.route('/train_model', methods=['POST'])
def train_model():
    print('Received form data:', request.form)
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract model parameters and training options from the request
        model_params = request.form.get('model_params', '{}')
        model_params = json.loads(model_params)
        
        model_type = request.form.get('model_type', None)
        target_column = request.form.get('target_column', None)    
        # Read the file into a DataFrame if it's a CSV
        if filename.rsplit('.', 1)[1].lower() == 'csv':
            data = pd.read_csv(filepath)
            if not target_column or target_column not in data.columns:
                return jsonify({"error": "Target column not specified or not found"}), 400
            
            # Identify feature columns (excluding the target column and identifier columns)
            feature_columns = [col for col in data.columns if col != target_column and not is_identifier_column(col)]
            
            auto_input_size = 'auto_input_size' in request.form
            auto_output_size = 'auto_output_size' in request.form
            auto_input_shape = 'auto_input_shape' in request.form

            if auto_input_size or request.form.get('input_size') == 'auto':
                input_size = data[feature_columns].shape[1]
            else:
                input_size = int(request.form.get('input_size'))
            
            if auto_output_size or request.form.get('output_size') == 'auto':
                output_size = len(data[target_column].unique())
            else:
                output_size = int(request.form.get('output_size'))
            
            if auto_input_shape or request.form.get('input_shape') == 'auto':
                input_shape = [data[feature_columns].shape[1]]
            else:
                input_shape = json.loads(request.form.get('input_shape'))

            model_params = {}
            if model_type == 'pytorch':
                model_params['input_size'] = input_size
                model_params['hidden_size'] = int(request.form.get('hidden_size'))
                model_params['output_size'] = output_size
                model_params['learning_rate'] = float(request.form.get('learning_rate'))
                model_params['batch_size'] = int(request.form.get('batch_size'))
                model_params['epochs'] = int(request.form.get('epochs'))
            elif model_type == 'tensorflow':
                model_params['input_shape'] = input_shape
                model_params['learning_rate'] = float(request.form.get('learning_rate'))
                model_params['batch_size'] = int(request.form.get('batch_size'))
                model_params['epochs'] = int(request.form.get('epochs'))
            elif model_type == 'sklearn':
                model_params['test_size'] = float(request.form.get('test_size'))
                model_params['random_state'] = int(request.form.get('random_state'))
                model_params['max_iter'] = int(request.form.get('max_iter'))
            else:
                return jsonify({"error": "Invalid model type specified"}), 400
            

            job_data = {
                'X': data[feature_columns].values.tolist(),  # Keep only feature columns
                'y': data[target_column].values.tolist()               
            }
            if auto_input_size:
                model_params['input_size'] = data[feature_columns].shape[1]
            if auto_output_size:
                model_params['output_size'] = len(data[target_column].unique())
            if auto_input_shape:
                model_params['input_shape'] = [data[feature_columns].shape[1]]
        else:
            # Add logic for other file types here
            return jsonify({"error": "Unsupported file type"}), 400
        # Determine which training function to call based on the model type
        if model_type == 'pytorch':
            task = celery.send_task('app.ml.train_pytorch_model', args=[job_data, model_params])
        elif model_type == 'tensorflow':
            task = celery.send_task('app.ml.train_tensorflow_model', args=[job_data, model_params])
        elif model_type == 'sklearn':
            task = celery.send_task('app.ml.train_sklearn_model', args=[job_data, model_params])
        else:
            return jsonify({"error": "Invalid model type specified"}), 400
        
        return jsonify({"message": "Job submitted successfully", "job_id": task.id}), 202
    else:
        return jsonify({"error": "File type not allowed"}), 400


@main.route('/job_status/<job_id>', methods=['GET'])
def job_status(job_id):
    try:
        task = AsyncResult(job_id, app=celery)
        response = {
            'job_id': job_id,
            'status': task.status,
            'result': task.result if task.ready() else None
        }
        return jsonify(response), 200
    except KeyError as e:
        # Handle the KeyError specifically or return a custom error message
        return jsonify({'error': f'Key not found: {e.args[0]}'}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({'error': str(e)}), 500

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
    

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
