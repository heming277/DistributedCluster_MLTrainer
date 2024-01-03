# app/__init__.py
from flask import Flask
from app.celery_worker import init_celery

def create_app():
    app = Flask(__name__)
    # Configure Celery
    app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
    
    # Initialize Celery
    init_celery(app)

    from app.views import main as main_blueprint
    app.register_blueprint(main_blueprint)

    return app
