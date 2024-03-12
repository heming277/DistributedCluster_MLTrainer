# app/views.py
from flask import Blueprint, request, jsonify, current_app
from flask import render_template, send_from_directory
from flask import redirect, url_for, session, abort
from functools import wraps
from datetime import datetime
from app import oauth
from urllib.parse import quote_plus, urlencode
from werkzeug.utils import secure_filename
import os
import pandas as pd
import json
from .utils import get_job_status, get_job_result
from app.ml import train_pytorch_model, train_tensorflow_model, train_sklearn_model, plot_training_history
from app.celery_worker import celery
from celery.result import AsyncResult
from sklearn.impute import SimpleImputer
import numpy as np
import re
import hashlib



main = Blueprint('main', __name__)

@main.route('/')
def home():
    print(session) 
    return render_template('index.html', session=session.get('user'), pretty=json.dumps(session.get('user'), indent=4))


@main.route("/callback", methods=["GET", "POST"])
def callback():
    token = oauth.auth0.authorize_access_token()
    session["user"] = token
    return redirect("/")



@main.route("/login")
def login():
    return oauth.auth0.authorize_redirect(
        redirect_uri=url_for("main.callback", _external=True)
    )

@main.route('/signup')
def signup():
    # Redirect to Auth0 for signup
    redirect_uri = url_for('main.callback', _external=True)
    return oauth.auth0.authorize_redirect(redirect_uri=redirect_uri, screen_hint='signup')


@main.route("/logout")
def logout():
    session.clear()
    return redirect(
        "https://" + os.getenv("AUTH0_DOMAIN")
        + "/v2/logout?"
        + urlencode(
            {
                "returnTo": url_for("main.home", _external=True),
                "client_id": os.getenv("AUTH0_CLIENT_ID"),
            },
            quote_via=quote_plus,
        )
    )


def is_identifier_column(col_name):
    # Normalize the column name by removing whitespace and special characters
    normalized_col_name = re.sub(r'\W+', '', col_name).lower()
    # Check if the normalized column name matches common identifier patterns
    return normalized_col_name in ['id', 'index']


def file_hash(file_stream):
    """Generate a hash for a file stream."""
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: file_stream.read(4096), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()

@main.route('/train_model', methods=['POST'])
def train_model():
    filename = None
    print('Received form data:', request.form)
    file_reference = request.form.get('file_reference')

    file = request.files.get('file')
    if file and file.filename:
        if allowed_file(file.filename):
            file.stream.seek(0)  # Reset file stream position
            file_hash_value = file_hash(file.stream)
            filename = secure_filename(f"{file_hash_value}_{file.filename}")
            filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

            if not os.path.exists(filepath):
                file.stream.seek(0)  # Reset file stream position again before saving
                file.save(filepath)
            # Use this new filename as the file_reference for response
            file_reference = filename
        else:
            return jsonify({"error": "File type not allowed"}), 400
    elif file_reference:
        # No file uploaded, but file_reference is provided
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], file_reference)
        if not os.path.exists(filepath):
            return jsonify({"error": "File reference not found"}), 400
        filename = file_reference
    else:
        return jsonify({"error": "No file part or file reference provided"}), 400

        # Extract model parameters and training options from the request
    model_type = request.form.get('model_type', None)
    target_column = request.form.get('target_column', None)
        
    if filename.rsplit('.', 1)[1].lower() == 'csv':
        data = pd.read_csv(filepath, sep=None, engine='python') # Make sure to use the correct separator
        print("Columns in the CSV file:", data.columns.tolist())
        data = data.dropna(axis=1, how='all')
        if not target_column or target_column not in data.columns:
            return jsonify({
                "error": f"Target column '{target_column}' not specified or not found in {data.columns.tolist()}",
                "file_uploaded": True,
                "file_reference": file_reference
            }), 400
        
        feature_columns = [col for col in data.columns if col != target_column and not is_identifier_column(col)]
        
        # Initialize model_params dictionary
        model_params = {}

        if model_type == 'pytorch':
            # For PyTorch, set input_size and output_size
            if 'auto_input_size' in request.form or request.form.get('input_size') == 'auto':
                input_size = data[feature_columns].shape[1]
            else:
                input_size = int(request.form.get('input_size'))
            
            if 'auto_output_size' in request.form or request.form.get('output_size') == 'auto':
                output_size = len(data[target_column].unique())
            else:
                output_size = int(request.form.get('output_size'))
            
            model_params.update({
                'input_size': input_size,
                'output_size': output_size,
                'hidden_size': int(request.form.get('hidden_size')),
                'learning_rate': float(request.form.get('learning_rate')),
                'batch_size': int(request.form.get('batch_size')),
                'epochs': int(request.form.get('epochs'))
            })

        elif model_type == 'tensorflow':
            # For TensorFlow, set input_shape
            if 'auto_input_shape' in request.form or request.form.get('input_shape') == 'auto':
                input_shape = [data[feature_columns].shape[1]]
            else:
                input_shape = json.loads(request.form.get('input_shape'))
            
            model_params.update({
                'input_shape': input_shape,
                'learning_rate': float(request.form.get('learning_rate')),
                'batch_size': int(request.form.get('batch_size')),
                'epochs': int(request.form.get('epochs'))
            })

        elif model_type == 'sklearn':
            # For scikit-learn, set relevant parameters
            model_params.update({
                'test_size': float(request.form.get('test_size')),
                'random_state': int(request.form.get('random_state')),
                'max_iter': int(request.form.get('max_iter'))
            })

        else:
            return jsonify({"error": "Invalid model type specified"}), 400
        
        job_data = {
            'X': data[feature_columns].values.tolist(),
            'y': data[target_column].values.tolist()
        }

        # Send task to Celery based on model_type
        task = celery.send_task(f'app.ml.train_{model_type}_model', args=[job_data, model_params])

        response_data = {
            "message": "Job submitted successfully",
            "job_id": task.id,
            "file_reference": file_reference  # Assuming file_reference is meaningful for later retrieval or reference
        }
        return jsonify(response_data), 202
        #return jsonify({"message": "Job submitted successfully", "job_id": task.id, "file_reference": file_reference}), 202
    else:
        return jsonify({"error": "Unsupported file type"}), 400






@main.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)

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
        # Handle KeyError
        return jsonify({'error': f'Key not found: {e.args[0]}'}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({'error': str(e)}), 500

@main.route('/job_result/<job_id>', methods=['GET'])
def job_result(job_id):
    print(f"Job result requested for jobId: {job_id}") 
    task = AsyncResult(job_id, app=celery)
    if task.ready():
        task_result = task.result
        if 'model_filename' in task_result:
            model_filename = task_result['model_filename']
            model_save_path = os.path.join(current_app.config['MODELS_DIRECTORY'], model_filename)
            
            # Check if history data is available for plotting
            if 'history' in task_result:
                history = task_result['history']
                plot_filename = f"training_history_{task_result['model_type']}.png"
                plot_path = os.path.join(current_app.config['UPLOAD_FOLDER'], plot_filename)
                
                if task_result['model_type'] == 'tensorflow':
                    # Ensure history is in the correct format for TensorFlow cases
                    adjusted_history = {
                        'train_accuracy': history['accuracy'],
                        'val_accuracy': history['val_accuracy'],
                        'train_loss': history['loss'],
                        'val_loss': history['val_loss']
                    }
                    plot_training_history(adjusted_history, plot_path)
                else:
                    # For PyTorch or other model types, assume history is already in the correct format
                    plot_training_history(history, plot_path)
                
                # Include plot filename in the response
                response_data = {
                    'job_id': job_id,
                    'status': task.status,
                    'result': task_result,
                    'model_save_path': model_save_path,
                    'plot_filename': plot_filename
                }
            else:
                response_data = {
                    'job_id': job_id,
                    'status': task.status,
                    'result': task_result,
                    'model_save_path': model_save_path
                }
            return jsonify(response_data), 200
        else:
            # Handle the case where model_filename is not set
            response = {
                'job_id': job_id,
                'error': 'Model filename is not available.',
                'status': task.status
            }
            return jsonify(response), 500
    else:
        response = {
            'job_id': job_id,
            'error': 'Result not found or job still processing',
            'status': task.status
        }
        return jsonify(response), 404







@main.route('/download/<path:filename>', methods=['GET'])  
def download(filename):
    models_directory = current_app.config['MODELS_DIRECTORY']
    try:
        # Change 'filename=filename' to 'path=filename'
        return send_from_directory(directory=models_directory, path=filename, as_attachment=True)
    except FileNotFoundError:
        abort(404)



def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'profile' not in session:
            # Redirect to Login page here
            return redirect('/')
        return f(*args, **kwargs)

    return decorated