
        let selectedModel = null;
        let selectedInputType = null;
        let selectedFile = null;
        let quizAnswers = {};
        
        // Question sets (will be loaded from backend)
        let ARMYWORM_QUESTIONS = [];
        let DISEASE_QUESTIONS = [];

        // Initialize page
        window.onload = function() {
            checkModelStatus();
            loadQuizQuestions();
            setupModeSelection();
            setupFileUpload();
            setupDragDrop();
        };

        function checkModelStatus() {
            fetch('/model_status')
                .then(response => response.json())
                .then(data => {
                    Object.keys(data).forEach(model => {
                        const statusIndicator = document.getElementById(`status-${model.replace('_', '-')}`);
                        const statusIndicatorQuiz = document.getElementById(`status-${model.replace('_', '-')}-quiz`);
                        
                        if (statusIndicator) {
                            statusIndicator.className = `status-indicator ${data[model] ? 'status-loaded' : 'status-not-loaded'}`;
                        }
                        if (statusIndicatorQuiz) {
                            statusIndicatorQuiz.className = `status-indicator ${data[model] ? 'status-loaded' : 'status-not-loaded'}`;
                        }
                    });
                })
                .catch(error => {
                    console.error('Error checking model status:', error);
                });
        }

        function loadQuizQuestions() {
            fetch('/quiz_questions')
                .then(response => response.json())
                .then(data => {
                    ARMYWORM_QUESTIONS = data.armyworm_questions || [];
                    DISEASE_QUESTIONS = data.disease_questions || [];
                })
                .catch(error => {
                    console.error('Error loading quiz questions:', error);
                });
        }

        function setupModeSelection() {
            document.querySelectorAll('.model-card').forEach(card => {
                card.addEventListener('click', function() {
                    // Remove previous selections
                    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
                    
                    // Select current card
                    this.classList.add('selected');
                    selectedModel = this.dataset.model;
                    selectedInputType = this.dataset.input;
                    
                    // Hide all sections first
                    document.getElementById('uploadSection').classList.remove('show');
                    document.getElementById('quizSection').classList.remove('show');
                    document.getElementById('results').classList.remove('show');
                    
                    // Show appropriate section
                    if (selectedInputType === 'image' || selectedInputType === 'data') {
                        showUploadSection();
                    } else if (selectedInputType === 'quiz') {
                        showQuizSection();
                    }
                });
            });
        }

        function showUploadSection() {
            document.getElementById('uploadSection').classList.add('show');
            
            const uploadTitle = document.getElementById('uploadTitle');
            const instructions = document.getElementById('uploadInstructions');
            const fileInput = document.getElementById('fileInput');
            
            if (selectedInputType === 'image') {
                uploadTitle.textContent = selectedModel.includes('disease') ? 
                    'Upload Plant Disease Images' : 'Upload Crop Images for Insect Detection';
                instructions.textContent = 'Upload image files (PNG, JPG, JPEG, GIF)';
                fileInput.accept = '.png,.jpg,.jpeg,.gif';
            } else if (selectedInputType === 'data') {
                uploadTitle.textContent = selectedModel.includes('disease') ? 
                    'Upload Disease Analysis Data' : 'Upload Insect Analysis Data';
                instructions.textContent = 'Upload data files (CSV, Excel) - should have 30 feature columns';
                fileInput.accept = '.csv,.xlsx,.xls';
            }
            
            // Reset file selection
            selectedFile = null;
            document.getElementById('fileInfo').classList.remove('show');
            document.getElementById('predictBtn').disabled = true;
        }

        function showQuizSection() {
            document.getElementById('quizSection').classList.add('show');
            generateQuiz();
        }

        function generateQuiz() {
            const questions = selectedModel === 'insect_tabnet' ? ARMYWORM_QUESTIONS : DISEASE_QUESTIONS;
            const title = selectedModel === 'insect_tabnet' ? 
                '🔍 Armyworm Detection Assessment' : '🦠 Plant Disease Assessment';
            
            document.getElementById('quizTitle').textContent = title;
            
            const container = document.getElementById('questionsContainer');
            container.innerHTML = '';
            
            questions.forEach((question, index) => {
                const questionCard = document.createElement('div');
                questionCard.className = 'question-card';
                questionCard.innerHTML = `
                    <div class="question-number">${index + 1}</div>
                    <div class="question-text">${question}</div>
                    <div class="answer-options">
                        <button class="answer-btn" data-question="${index}" data-answer="1" onclick="selectAnswer(${index}, 1, this)">
                            ✅ Yes
                        </button>
                        <button class="answer-btn" data-question="${index}" data-answer="0" onclick="selectAnswer(${index}, 0, this)">
                            ❌ No
                        </button>
                    </div>
                `;
                container.appendChild(questionCard);
            });
            
            resetQuiz();
        }

        function selectAnswer(questionIndex, answer, buttonElement) {
            // Remove previous selection for this question
            const questionCard = buttonElement.closest('.question-card');
            questionCard.querySelectorAll('.answer-btn').forEach(btn => btn.classList.remove('selected'));
            
            // Add selection to clicked button
            buttonElement.classList.add('selected');
            
            // Store answer
            quizAnswers[questionIndex] = answer;
            
            updateQuizProgress();
        }

        function updateQuizProgress() {
            const totalQuestions = selectedModel === 'insect_tabnet' ? ARMYWORM_QUESTIONS.length : DISEASE_QUESTIONS.length;
            const answeredCount = Object.keys(quizAnswers).length;
            const progressPercent = Math.round((answeredCount / totalQuestions) * 100);
            
            document.getElementById('answeredCount').textContent = answeredCount;
            document.getElementById('progressPercent').textContent = progressPercent + '%';
            document.getElementById('progressBar').style.width = progressPercent + '%';
            
            // Enable submit button if all questions answered
            document.getElementById('submitQuizBtn').disabled = answeredCount < totalQuestions;
        }

        function resetQuiz() {
            quizAnswers = {};
            document.querySelectorAll('.answer-btn').forEach(btn => btn.classList.remove('selected'));
            updateQuizProgress();
        }

        function setupFileUpload() {
            const fileInput = document.getElementById('fileInput');
            const predictBtn = document.getElementById('predictBtn');
            
            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    selectedFile = file;
                    document.getElementById('fileName').textContent = file.name;
                    document.getElementById('fileInfo').classList.add('show');
                    predictBtn.disabled = false;
                }
            });
            
            predictBtn.addEventListener('click', function() {
                if (selectedFile && selectedModel && selectedInputType) {
                    submitFilePrediction();
                }
            });
        }

        function setupDragDrop() {
            const uploadSection = document.getElementById('uploadSection');
            
            uploadSection.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.classList.add('dragover');
            });
            
            uploadSection.addEventListener('dragleave', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
            });
            
            uploadSection.addEventListener('drop', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
                
                const file = e.dataTransfer.files[0];
                if (file && selectedModel) {
                    selectedFile = file;
                    document.getElementById('fileName').textContent = file.name;
                    document.getElementById('fileInfo').classList.add('show');
                    document.getElementById('predictBtn').disabled = false;
                }
            });
        }

        function submitFilePrediction() {
            if (!selectedFile || !selectedModel || !selectedInputType) {
                showError('Please select a model and upload a file');
                return;
            }

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('model_type', selectedModel);
            formData.append('input_type', selectedInputType);

            showLoading();

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    showError(data.error);
                } else {
                    showResults(data);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error.message);
            });
        }

        function submitQuiz() {
            const totalQuestions = selectedModel === 'insect_tabnet' ? ARMYWORM_QUESTIONS.length : DISEASE_QUESTIONS.length;
            
            if (Object.keys(quizAnswers).length < totalQuestions) {
                showError('Please answer all questions before submitting');
                return;
            }

            // Convert answers to the format expected by the backend
            const answersArray = [];
            for (let i = 0; i < totalQuestions; i++) {
                answersArray.push(quizAnswers[i] || 0);
            }

            const requestData = {
                model_type: selectedModel,
                input_type: 'quiz',
                answers: answersArray
            };

            showLoading();

            fetch('/predict_quiz', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.error) {
                    showError(data.error);
                } else {
                    showResults(data);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error.message);
            });
        }

        function showResults(data) {
            const resultContent = document.getElementById('resultContent');
            let html = '';

            html += `<div class="result-item">
                <span class="result-label">Model Used:</span>
                <span class="result-value">${data.model_used}</span>
            </div>`;

            html += `<div class="result-item">
                <span class="result-label">Input Type:</span>
                <span class="result-value">${data.input_type || selectedInputType}</span>
            </div>`;

            // Handle YOLO detection results (Image input)
            if (data.predictions && Array.isArray(data.predictions)) {
                html += `<div class="prediction-highlight">
                    <div class="predicted-class">Detections Found: ${data.detection_count}</div>
                    <div class="predicted-confidence">Average Confidence: ${(data.avg_confidence * 100).toFixed(2)}%</div>
                </div>`;

                // Show result image if available
                if (data.result_image) {
                    html += `<div class="result-item">
                        <span class="result-label">Detection Results:</span>
                        <br>
                        <img src="${data.result_image}" alt="Detection Results" class="result-image">
                    </div>`;
                }

                // Show individual detections
                if (data.predictions.length > 0) {
                    html += `<div class="result-item">
                        <span class="result-label">Individual Detections:</span>
                        <div class="detection-grid">`;
                    
                    data.predictions.forEach((detection, index) => {
                        html += `<div class="detection-item">
                            <div class="detection-class">${detection.class}</div>
                            <div class="detection-confidence">Confidence: ${(detection.confidence * 100).toFixed(2)}%</div>
                            <div class="bbox-info">BBox: [${detection.bbox.map(x => x.toFixed(1)).join(', ')}]</div>
                        </div>`;
                    });
                    
                    html += `</div></div>`;
                }
            }
            // Handle TabNet results (Quiz or CSV input)
            else if (data.prediction !== undefined) {
                // Determine prediction text based on model and result
                let predictionText = '';
                let predictionClass = '';
                
                if (selectedModel === 'insect_tabnet') {
                    const prediction = Array.isArray(data.prediction) ? data.prediction[0] : data.prediction;
                    predictionText = prediction === 1 ? 'Armyworm Detected' : 'No Armyworm Detected';
                    predictionClass = prediction === 1 ? '🐛 Positive Detection' : '✅ Negative Detection';
                } else if (selectedModel === 'disease_tabnet') {
                    const prediction = Array.isArray(data.prediction) ? data.prediction[0] : data.prediction;
                    predictionText = prediction === 1 ? 'Disease Detected' : 'Healthy Plant';
                    predictionClass = prediction === 1 ? '🦠 Disease Present' : '🌱 Healthy Plant';
                }
                
                html += `<div class="prediction-highlight">
                    <div class="predicted-class">${predictionClass}</div>
                    <div class="predicted-confidence">${predictionText}</div>`;
                    
                // Show confidence for single prediction
                if (data.probabilities && Array.isArray(data.probabilities)) {
                    const probs = Array.isArray(data.probabilities[0]) ? data.probabilities[0] : data.probabilities;
                    const maxProb = Math.max(...probs);
                    html += `<div style="margin-top: 10px; font-size: 1em;">Confidence: ${(maxProb * 100).toFixed(2)}%</div>`;
                }
                
                html += `</div>`;

                // Show additional info for CSV input
                if (data.input_type === 'data' || data.data_shape) {
                    html += `<div class="result-item">
                        <span class="result-label">Data Processed:</span>
                        <span class="result-value">${data.num_samples || 'N/A'} samples</span>
                    </div>`;
                    
                    if (data.data_shape) {
                        html += `<div class="result-item">
                            <span class="result-label">Data Shape:</span>
                            <span class="result-value">${data.data_shape[0]} rows × ${data.data_shape[1]} columns</span>
                        </div>`;
                    }
                }

                // Show probabilities if available
                if (data.probabilities) {
                    html += `<div class="probability-container">
                        <h4>Class Probabilities</h4>`;
                    
                    // Handle both single predictions and batch predictions
                    if (Array.isArray(data.probabilities[0]) && data.probabilities.length > 1) {
                        // Batch predictions - show summary
                        html += `<div class="result-item">
                            <span class="result-label">Batch Processing:</span>
                            <span class="result-value">${data.probabilities.length} samples processed</span>
                        </div>`;
                        
                        // Show average probabilities
                        const avgProbs = data.probabilities[0].map((_, i) => {
                            const sum = data.probabilities.reduce((acc, probs) => acc + (probs[i] || 0), 0);
                            return sum / data.probabilities.length;
                        });
                        
                        html += `<h5>Average Probabilities:</h5>`;
                        avgProbs.forEach((prob, index) => {
                            const label = selectedModel === 'insect_tabnet' 
                                ? (index === 0 ? 'No Armyworm' : 'Armyworm Present')
                                : (index === 0 ? 'Healthy' : 'Disease Present');
                            
                            html += `<div class="probability-bar">
                                <span class="probability-class">${label}</span>
                                <div class="probability-progress">
                                    <div class="probability-progress-fill" style="width: ${prob * 100}%"></div>
                                </div>
                                <span class="probability-percentage">${(prob * 100).toFixed(2)}%</span>
                            </div>`;
                        });
                    } else {
                        // Single prediction
                        const probs = Array.isArray(data.probabilities[0]) ? data.probabilities[0] : data.probabilities;
                        const labels = selectedModel === 'insect_tabnet' 
                            ? ['No Armyworm', 'Armyworm Present']
                            : ['Healthy', 'Disease Present'];
                        
                        probs.forEach((prob, index) => {
                            html += `<div class="probability-bar">
                                <span class="probability-class">${labels[index] || `Class ${index}`}</span>
                                <div class="probability-progress">
                                    <div class="probability-progress-fill" style="width: ${prob * 100}%"></div>
                                </div>
                                <span class="probability-percentage">${(prob * 100).toFixed(2)}%</span>
                            </div>`;
                        });
                    }
                    
                    html += `</div>`;
                }
            }

            resultContent.innerHTML = html;
            document.getElementById('results').classList.add('show');
            
            // Animate progress bars
            setTimeout(() => {
                document.querySelectorAll('.probability-progress-fill, .confidence-fill').forEach(fill => {
                    const width = fill.style.width;
                    fill.style.width = '0%';
                    setTimeout(() => {
                        fill.style.width = width;
                    }, 100);
                });
            }, 100);
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            
            // Auto-hide error after 5 seconds
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }

        function showLoading() {
            document.getElementById('loading').style.display = 'block';
            document.getElementById('results').classList.remove('show');
            document.getElementById('error').style.display = 'none';
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }