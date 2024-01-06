# app/__init__.py
from flask import Flask
import os
from dotenv import load_dotenv
from app.ml import train_pytorch_model, train_tensorflow_model, train_sklearn_model

# Load environment variables from .env file
load_dotenv()

def create_app():
    app = Flask(__name__)
    # Configure Celery using environment variables
    app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

    # Set the upload folder for file uploads
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')

    # Ensure the upload path exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize Celery
    from app.celery_worker import init_celery
    init_celery(app)

    

    from app.views import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app