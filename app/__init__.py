# app/__init__.py
from flask import Flask
import os
from dotenv import load_dotenv
from authlib.integrations.flask_client import OAuth
from flask_cors import CORS

oauth = OAuth()
load_dotenv()


def create_app():
    app = Flask(__name__)
    CORS(app)
    # Configure Celery using environment variables
    app.secret_key = os.getenv("APP_SECRET_KEY")
    app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://redis:6379/0')
    app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://redis:6379/0')

    # Set the upload folder for file uploads
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
    # Ensure the upload path exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Configure the models directory
    app.config['MODELS_DIRECTORY'] = os.path.join(BASE_DIR, 'models')
    # Ensure the models directory exists
    os.makedirs(app.config['MODELS_DIRECTORY'], exist_ok=True)
    
    # Initialize Celery
    from app.celery_worker import init_celery
    init_celery(app)

    # Configure the OAuth object with Auth0
    oauth.init_app(app)
    
    oauth.register(
        "auth0",
        client_id=os.getenv("AUTH0_CLIENT_ID"),
        client_secret=os.getenv("AUTH0_CLIENT_SECRET"),
        client_kwargs={
            "scope": "openid profile email",
        },
        server_metadata_url=f'https://{os.getenv("AUTH0_DOMAIN")}/.well-known/openid-configuration'
    )

    # Register the main blueprint
    from app.views import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
