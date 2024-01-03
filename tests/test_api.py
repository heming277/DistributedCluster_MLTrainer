# tests/test_api.py
import pytest
from app import create_app

@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    return app

@pytest.fixture
def client(app):
    return app.test_client()

def test_hello_world(client):
    response = client.get('/')
    assert response.data == b'Hello, World!'
    assert response.status_code == 200


def test_submit_job(client):
    job_data = {
        'model_type': 'pytorch',  # or 'tensorflow'/'sklearn'
        # ... include other necessary data for the job
    }
    response = client.post('/submit_job', json=job_data)
    assert response.status_code == 202
    assert 'job_id' in response.json

def test_job_status(client):
    #  valid job_id from a previously submitted job
    job_id = 'some-valid-job-id'
    response = client.get(f'/job_status/{job_id}')
    assert response.status_code == 200
    assert response.json['job_id'] == job_id
    # ... additional assertions based on expected job status



def test_submit_and_check_job_status(client):
    job_data = {
        'model_type': 'pytorch',  # or 'tensorflow'/'sklearn'
        # ... include other necessary data for the job
    }
    submit_response = client.post('/submit_job', json=job_data)
    assert submit_response.status_code == 202
    job_id = submit_response.json['job_id']
    print(f"Submitted job with ID: {job_id}")


    # Now that we have a valid job_id, we can check its status
    status_response = client.get(f'/job_status/{job_id}')
    assert status_response.status_code == 200
    assert status_response.json['job_id'] == job_id
    # ... additional assertions based on expected job status
    print(f"Job status response: {status_response.json}")
