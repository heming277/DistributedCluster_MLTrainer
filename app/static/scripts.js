// frontend/scripts.js
document.getElementById('upload-form').addEventListener('submit', function(e) {
    e.preventDefault();
    const formData = new FormData(this);

    if (document.getElementById('auto_input_size').checked) {
        formData.set('input_size', 'auto');
    }
    if (document.getElementById('auto_output_size').checked) {
        formData.set('output_size', 'auto');
    }
    if (document.getElementById('auto_input_shape').checked) {
        formData.set('input_shape', 'auto');
    }
    console.log('auto_input_size checked:', document.getElementById('auto_input_size').checked);
    console.log('auto_output_size checked:', document.getElementById('auto_output_size').checked);
    console.log('auto_input_shape checked:', document.getElementById('auto_input_shape').checked);
    
    // Log formData for debugging
    for (let [key, value] of formData.entries()) {
        console.log(key, value);
    }


    fetch('/train_model', {
        method: 'POST',
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        if (data.job_id) {
            document.getElementById('status-container').style.display = 'block';
            checkJobStatus(data.job_id);
        } else {
            alert(data.error || 'Error occurred');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while uploading the file.');
    });
});

document.addEventListener('DOMContentLoaded', function() {
    const modelTypeSelect = document.getElementById('model_type');

    const autoInputSizeContainer = document.getElementById('auto_input_size_container');
    const autoOutputSizeContainer = document.getElementById('auto_output_size_container');
    const autoInputShapeContainer = document.getElementById('auto_input_shape_container');

    function updateFormFields(modelType) {
        const modelParamsContainer = document.getElementById('model_params_container');
        modelParamsContainer.innerHTML = ''; // Clear previous parameters

        // Hide all auto-detect containers by default
        autoInputSizeContainer.style.display = 'none';
        autoOutputSizeContainer.style.display = 'none';
        autoInputShapeContainer.style.display = 'none';

        let paramsHtml = '';
        if (modelType === 'pytorch') {
            autoInputSizeContainer.style.display = 'block';
            autoOutputSizeContainer.style.display = 'block';
            paramsHtml += createParamInput('input_size', 'Input Size (Number of input features)', 'e.g., 4');
            paramsHtml += createParamInput('hidden_size', 'Hidden Size (Number of neurons in hidden layer)', 'e.g., 10');
            paramsHtml += createParamInput('output_size', 'Output Size (Number of output classes)', 'e.g., 3');
            paramsHtml += createParamInput('learning_rate', 'Learning Rate (Step size for updating weights)', 'e.g., 0.01');
            paramsHtml += createParamInput('batch_size', 'Batch Size (Number of samples per gradient update)', 'e.g., 16');
            paramsHtml += createParamInput('epochs', 'Epochs (Number of passes over the entire dataset)', 'e.g., 10');
        } else if (modelType === 'tensorflow') {
            autoInputShapeContainer.style.display = 'block';
            paramsHtml += createParamInput('input_shape', 'Input Shape (Shape of the input data excluding batch size)', 'e.g., [4]');
            paramsHtml += createParamInput('learning_rate', 'Learning Rate (Step size for updating weights)', 'e.g., 0.01');
            paramsHtml += createParamInput('batch_size', 'Batch Size (Number of samples per gradient update)', 'e.g., 16');
            paramsHtml += createParamInput('epochs', 'Epochs (Number of passes over the entire dataset)', 'e.g., 10');
        
        } else if (modelType === 'sklearn') {
            paramsHtml += createParamInput('test_size', 'Test Size (Proportion of the dataset to include in the test split)', 'e.g., 0.2');
            paramsHtml += createParamInput('random_state', 'Random State (Seed used by the random number generator)', 'e.g., 42');
            paramsHtml += createParamInput('max_iter', 'Max Iterations (Maximum number of iterations taken for the solvers to converge)', 'e.g., 100');
        }
        modelParamsContainer.innerHTML = paramsHtml;
        toggleInputSizeField();
        toggleOutputSizeField();
        toggleInputShapeField();
    }

function toggleInputSizeField() {
    const isChecked = document.getElementById('auto_input_size').checked;
    const inputSizeField = document.querySelector('input[name="input_size"]');
    if (inputSizeField) {
        inputSizeField.parentElement.style.display = isChecked ? 'none' : 'block';
        // Update the required attribute based on the checkbox
        if (isChecked) {
            inputSizeField.removeAttribute('required');
        } else {
            inputSizeField.setAttribute('required', '');
        }
    }
}

function toggleOutputSizeField() {
    const isChecked = document.getElementById('auto_output_size').checked;
    const outputSizeField = document.querySelector('input[name="output_size"]');
    if (outputSizeField) {
        outputSizeField.parentElement.style.display = isChecked ? 'none' : 'block';
        // Update the required attribute based on the checkbox
        if (isChecked) {
            outputSizeField.removeAttribute('required');
        } else {
            outputSizeField.setAttribute('required', '');
        }
    }
}

function toggleInputShapeField() {
    const isChecked = document.getElementById('auto_input_shape').checked;
    const inputShapeField = document.querySelector('input[name="input_shape"]');
    if (inputShapeField) {
        inputShapeField.parentElement.style.display = isChecked ? 'none' : 'block';
        // Update the required attribute based on the checkbox
        if (isChecked) {
            inputShapeField.removeAttribute('required');
        } else {
            inputShapeField.setAttribute('required', '');
        }
    }
}


    // Event listener for model type selection
    modelTypeSelect.addEventListener('change', function(e) {
        updateFormFields(e.target.value);
    });

    // Event listeners for the auto-detect checkboxes
    document.getElementById('auto_input_size').addEventListener('change', toggleInputSizeField);
    document.getElementById('auto_output_size').addEventListener('change', toggleOutputSizeField);
    document.getElementById('auto_input_shape').addEventListener('change', toggleInputShapeField);

    // Initialize the form with the correct visibility of fields
    updateFormFields(modelTypeSelect.value);
});


function createParamInput(name, label, placeholder) {
    return `
        <div class="form-group">
            <label>${label}:</label>
            <input type="text" class="form-control" name="${name}" placeholder="${placeholder}" required>
        </div>
    `;
}



function checkJobStatus(jobId) {
    fetch(`/job_result/${jobId}`)
    .then(response => response.json())
    .then(data => {
        document.getElementById('job-status').textContent = data.status;
        if (data.status === 'SUCCESS') {
            // Job is successful, display results and download link
            if (data.result) {
                document.getElementById('job-result').textContent = JSON.stringify(data.result, null, 2);
            }
            // Show the download link
            const downloadLink = document.getElementById('download-link');
            downloadLink.href = `/download/${encodeURIComponent(data.result.model_filename)}`;
            document.getElementById('download-container').style.display = 'block';
        } else if (data.status === 'FAILURE') {
            // Job failed, display error message
            document.getElementById('job-result').textContent = 'Job failed. Please try again.';
        } else {
            // If the job is still running, check again after a delay
            setTimeout(() => checkJobStatus(jobId), 2000);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while checking the job status.');
    });
}


