# app/utils.py
from celery.result import AsyncResult
from app.celery_worker import celery


def get_job_status(job_id):
    task = AsyncResult(job_id, app=celery)
    return {
        'job_id': job_id,
        'status': task.status
    }

def get_job_result(job_id):
    task = AsyncResult(job_id, app=celery)
    if task.ready():
        return {
            'status': task.status,
            'result': task.result
        }
    else:
        return {
            'status': task.status,
            'result': None,
            'error': 'Result not found or job still processing'
        }
