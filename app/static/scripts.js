// frontend/scripts.js
document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file');
    const submitButton = document.querySelector('button[type="submit"]'); // 
    const fileRefInput = document.getElementById('file_reference'); //
    // Function to update the button text
    function updateButtonText(text) {
        submitButton.textContent = text;
    }
    // Initially, set the button text to "Upload and Train"
    updateButtonText("Upload and Train");
    // Clear the file reference input value when a new file is selected
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        document.getElementById('loadingSpinner').style.display = 'flex'; // Show spinner
        const formData = new FormData(this);
        if (fileInput.files.length > 0) {
            fileRefInput.value = '';
            formData.delete('file_reference'); // Also remove it from formData if it was added
        } // heck if we already have a file reference, and a new file hasn't been selected
        if (fileRefInput.value && fileInput.files.length === 0) {
            formData.append('file_reference', fileRefInput.value);
        } else {
            // If a new file is selected, clear the previous file reference
            fileRefInput.value = '';
        }
        // Add checks  auto-detection checkboxes
        if (document.getElementById('auto_input_size').checked) {
            formData.set('input_size', 'auto');
        }
        if (document.getElementById('auto_output_size').checked) {
            formData.set('output_size', 'auto');
        }
        if (document.getElementById('auto_input_shape').checked) {
            formData.set('input_shape', 'auto');
        }
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
            if (data.file_uploaded && data.file_reference) {
                // File was uploaded but there was an error with the form
                document.getElementById('file_reference').value = data.file_reference;
                updateButtonText("Train"); // Change button text to indicate file is already uploaded
                alert(data.error); // Inform the user about the error
            } else if (data.job_id) {
                document.getElementById('status-container').style.display = 'block';
                checkJobStatus(data.job_id);

                if (data.file_reference) {
                    fileRefInput.value = data.file_reference;
                    // Update the button text to "Train" after successful upload
                    updateButtonText("Train");
                }
            } else {
                alert(data.error || 'Error occurred');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while uploading the file.');
        });
    });
    // Event listener for file input change
    fileInput.addEventListener('change', function() {
        fileRefInput.value = '';
        // Change the button text back to "Upload and Train" if a new file is selected
        updateButtonText("Upload and Train");
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

function fetchTrainingResult(jobId) {
    fetch(`/job_result/${jobId}`)
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        if (data.status === 'SUCCESS' && data.result) {
            displayModelResults(data.result);
            // Update the plot image within the modal
            const plotElement = document.getElementById('trainingPlot');
            if (data.plot_filename) {
                const timestamp = new Date().getTime();
                const plotImgUrl = `/uploads/${encodeURIComponent(data.plot_filename)}?t=${timestamp}`;
                plotElement.src = plotImgUrl;
                plotElement.style.display = 'block'; // Ensure the image is visible
            }
            else {
                plotElement.style.display = 'none'; // Hide the plot if no filename is provided
                plotElement.src = ''; // Optionally reset the src attribute
            }
            showModal(); // This function shows the modal, which now includes the plot
        } else {
            document.getElementById('modal-job-result').textContent = 'Job completed, but no results found.';
            showModal();
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while fetching the training result.');
    });
}


function displayModelResults(data) {
    const modalJobStatus = document.getElementById('modal-job-status');
    const modalJobResult = document.getElementById('modal-job-result');
    const modalDownloadLink = document.getElementById('modal-download-link');
    const modalDownloadContainer = document.getElementById('modal-download-container');
  
    // Update job status to 'SUCCESS'
    modalJobStatus.textContent = 'SUCCESS';

    // Initialize resultText with the message from the response
    let resultText = `${data.message}\n\n`;

    // Check and append train accuracy if available
    if (data.train_accuracy !== undefined) {
        resultText += `Train Accuracy: ${data.train_accuracy.toFixed(4)}\n`;
    }

    // Check and append validation accuracy if available
    if (data.val_accuracy !== undefined) {
        resultText += `Validation Accuracy: ${data.val_accuracy.toFixed(4)}\n`;
    } else if (data.validation_accuracy !== undefined) { // TensorFlow model uses 'validation_accuracy'
        resultText += `Validation Accuracy: ${data.validation_accuracy.toFixed(4)}\n`;
    }

    // Check and append accuracy if available (for scikit-learn model)
    if (data.accuracy !== undefined) {
        resultText += `Validation Accuracy: ${data.accuracy.toFixed(4)}\n`;
    }

    // Check and append validation loss if available
    if (data.validation_loss !== undefined) {
        resultText += `Validation Loss: ${data.validation_loss.toFixed(4)}\n`;
    }

    // Update the modal with the result text
    modalJobResult.textContent = resultText;
  
    // Update download link for the model
    modalDownloadLink.href = `/download/${encodeURIComponent(data.model_filename)}`;
    modalDownloadContainer.style.display = 'block';
}

function checkJobStatus(jobId) {
    console.log("Checking job status for jobId:", jobId);
    const requestUrl = `/job_result/${jobId}`;
    console.log("Request URL:", requestUrl);
    fetch(`/job_result/${jobId}`)
    .then(response => {
        if (response.status === 404) {
            console.log('Job result not ready yet, retrying...');
            // Schedule another check after a delay
            setTimeout(() => checkJobStatus(jobId), 2000);
            throw new Error('JobNotReady');// Return null to prevent further processing in this promise chain
        }
        if (!response.ok) {
            // For any other non-OK responses, throw an error to be caught by the catch block
            throw new Error('Network response was not ok: ' + response.statusText);
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('modal-job-status').textContent = data.status;
        if (data.status === 'SUCCESS') {
            // Directly handle the success logic here, avoiding an additional fetch
            displayModelResults(data.result);
            const plotElement = document.getElementById('trainingPlot');
            if (data.plot_filename) {
                const timestamp = new Date().getTime();
                const plotImgUrl = `/uploads/${encodeURIComponent(data.plot_filename)}?t=${timestamp}`;
                plotElement.src = plotImgUrl;
                plotElement.style.display = 'block';
            } else {
                plotElement.style.display = 'none';
                plotElement.src = '';
            }
            showModal();
        } else if (data.status === 'FAILURE') {
            document.getElementById('modal-job-result').textContent = 'Job failed. Please try again.';
            showModal();
        } else {
            setTimeout(() => checkJobStatus(jobId), 2000); // Poll if the job is still running
        }
    })
    .catch(error => {
        if (error.message === 'JobNotReady') {
            // This is our custom error for a not-ready job, so we don't need to log or alert
            return;
        }
        console.error('Error:', error);
        document.getElementById('loadingSpinner').style.display = 'none';
        alert('An error occurred while checking the job status.');
    });
}



// Function to show the modal
function showModal() {
    document.getElementById('loadingSpinner').style.display = 'none';
    var modal = document.getElementById("statusModal");
    modal.style.display = "block";
}

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close-button")[0];

// When the user clicks on <span> (x), close the modal
span.onclick = function() {
    var modal = document.getElementById("statusModal");
    modal.style.display = "none";
}



