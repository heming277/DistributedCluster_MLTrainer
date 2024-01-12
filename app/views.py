# app/views.py
from flask import Blueprint, request, jsonify, current_app
from flask import render_template, send_from_directory
from flask import redirect, url_for, session, abort
from functools import wraps
from app import oauth
from urllib.parse import quote_plus, urlencode
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
        model_type = request.form.get('model_type', None)
        target_column = request.form.get('target_column', None)
        
        if filename.rsplit('.', 1)[1].lower() == 'csv':
            data = pd.read_csv(filepath)
            if not target_column or target_column not in data.columns:
                return jsonify({"error": "Target column not specified or not found"}), 400
            
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
            return jsonify({"message": "Job submitted successfully", "job_id": task.id}), 202
        else:
            return jsonify({"error": "Unsupported file type"}), 400
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
        # Handle KeyError
        return jsonify({'error': f'Key not found: {e.args[0]}'}), 400
    except Exception as e:
        # Handle other exceptions
        return jsonify({'error': str(e)}), 500

@main.route('/job_result/<job_id>', methods=['GET'])
def job_result(job_id):
    task = AsyncResult(job_id, app=celery)
    if task.ready():
        print(task.result) 
        model_filename = task.result.get('model_filename')
        if model_filename is None:
            # Handle the case where model_filename is not set
            response = {
                'job_id': job_id,
                'error': 'Model filename is not available.',
                'status': task.status
            }
            return jsonify(response), 500
        model_save_path = os.path.join(current_app.config['MODELS_DIRECTORY'], model_filename)
        response = {
            'job_id': job_id,
            'status': task.status,
            'result': task.result,
            'model_save_path': model_save_path  # Add the model_save_path
        }
        return jsonify(response), 200
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