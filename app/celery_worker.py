# app/celery_worker.py
from celery import Celery
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

celery = Celery(__name__)

def init_celery(app):
    # Configure Celery using environment variables
    celery.conf.broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    celery.conf.result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    celery.conf.update(app.config)

    print('Celery broker URL:', celery.conf.broker_url)
    print('Celery result backend:', celery.conf.result_backend)

    class ContextTask(celery.Task):
        abstract = True
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return super(ContextTask, self).__call__(*args, **kwargs)

    celery.Task = ContextTask

# Import tasks after Celery instance has been created and configured
# Make sure the import path is correct based on your project structure
from app.ml import train_pytorch_model
from app.ml import train_tensorflow_model
from app.ml import train_sklearn_model