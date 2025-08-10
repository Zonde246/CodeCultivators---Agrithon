// Global variables
        let currentPipeline = null;
        let fusionTarget = null;
        let currentStep = 1;
        let quizAnswers = {};
        let selectedFile = null;
        let currentQuestionSet = [];
        let quizResult = null;
        let imageResult = null;
        let sessionId = null;
        
        // Individual analysis variables
        let individualModel = null;
        let individualInputType = null;
        let individualAnswers = {};
        let individualFile = null;
        let individualQuestionSet = [];
        let individualResult = null;
        
        // Question sets
        let DEAD_HEART_QUESTIONS = [];
        let TILLER_QUESTIONS = [];
        let DISEASE_QUESTIONS = [];

        // Question illustrations mapping
        const QUESTION_ILLUSTRATIONS = {
            'yellowing': '🟡', 'brown': '🟤', 'spots': '🔴', 'wilting': '🥀',
            'dried': '🍂', 'lesions': '🩹', 'rot': '💀', 'fungus': '🍄',
            'mold': '🦠', 'disease': '🦠', 'infected': '☣️', 'dead': '💀',
            'holes': '⚫', 'bore': '🕳️', 'tunnel': '🕳️', 'insect': '🐛',
            'pest': '🐛', 'caterpillar': '🐛', 'larva': '🐛', 'egg': '🥚',
            'leaves': '🍃', 'stem': '🌿', 'shoot': '🌱', 'tiller': '🌿',
            'water': '💧', 'rain': '🌧️', 'drought': '☀️', 'soil': '🌍'
        };

        // Initialize on page load
        window.onload = function() {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            setupPipelineSelection();
            setupIndividualAnalysis();
            loadQuizQuestions();
            setupFileUpload();
            setupIndividualFileUpload();
            setupDragDrop();
            setupIndividualDragDrop();
            checkModelStatus();
        };

        function getQuestionIllustration(questionText) {
            const text = questionText.toLowerCase();
            
            for (const [keyword, emoji] of Object.entries(QUESTION_ILLUSTRATIONS)) {
                if (text.includes(keyword)) {
                    return emoji;
                }
            }
            
            if (text.includes('see') || text.includes('notice') || text.includes('observe')) return '👁️';
            if (text.includes('count') || text.includes('number') || text.includes('many')) return '🔢';
            if (text.includes('color') || text.includes('colour')) return '🎨';
            if (text.includes('size') || text.includes('big') || text.includes('small')) return '📏';
            if (text.includes('time') || text.includes('when') || text.includes('day')) return '⏰';
            if (text.includes('weather') || text.includes('climate')) return '🌤️';
            
            return '🌾';
        }

        function checkModelStatus() {
            fetch('/model_status')
                .then(response => response.json())
                .then(data => {
                    updateModelStatusIndicators(data);
                })
                .catch(error => {
                    console.error('Error checking model status:', error);
                });
        }

        function updateModelStatusIndicators(status) {
            const indicators = {
                'status-disease-yolo': status.disease_yolo,
                'status-insect-yolo': status.insect_yolo,
                'status-disease-tabnet-quiz': status.disease_tabnet,
                'status-insect-tabnet-quiz': status.insect_tabnet
            };

            Object.entries(indicators).forEach(([id, loaded]) => {
                const element = document.getElementById(id);
                if (element) {
                    element.className = loaded ? 'status-indicator status-loaded' : 'status-indicator status-not-loaded';
                }
            });
        }

        function setupPipelineSelection() {
            document.querySelectorAll('.pipeline-card').forEach(card => {
                card.addEventListener('click', function() {
                    if (this.dataset.pipeline) {
                        selectPipeline(this.dataset.pipeline);
                    } else if (this.dataset.target) {
                        selectTarget(this.dataset.target);
                    }
                });
            });
        }

        function setupIndividualAnalysis() {
            document.querySelectorAll('#individualAnalysis .model-card').forEach(card => {
                card.addEventListener('click', function() {
                    handleIndividualSelection(this.dataset.model, this.dataset.input);
                });
            });
        }

        function handleIndividualSelection(model, inputType) {
            individualModel = model;
            individualInputType = inputType;
            
            document.querySelectorAll('#individualAnalysis .model-card').forEach(c => 
                c.classList.remove('selected'));
            
            event.target.closest('.model-card').classList.add('selected');
            
            document.getElementById('mainSelector').classList.add('hidden');
            document.getElementById('individualAnalysis').classList.add('hidden');
            document.getElementById('individualInterface').classList.add('show');
            
            if (inputType === 'image') {
                showIndividualImageUpload(model);
            } else if (inputType === 'quiz') {
                showIndividualQuiz(model);
            }
        }

        function showIndividualImageUpload(model) {
            document.getElementById('individualImageUpload').style.display = 'block';
            document.getElementById('individualQuiz').style.display = 'none';
            document.getElementById('individualResults').style.display = 'none';
            
            const title = model.includes('disease') ? '🦠 Disease Detection - Upload Image' : '🐛 Pest Detection - Upload Image';
            document.getElementById('individualImageTitle').textContent = title;
        }

        function showIndividualQuiz(model) {
            document.getElementById('individualImageUpload').style.display = 'none';
            document.getElementById('individualQuiz').style.display = 'block';
            document.getElementById('individualResults').style.display = 'none';
            
            const title = model.includes('disease') ? '🦠 Disease Assessment - Simple Questions' : '🐛 Pest Assessment - Simple Questions';
            document.getElementById('individualQuizTitle').textContent = title;
            
            generateIndividualQuiz(model);
        }

        function generateIndividualQuiz(model) {
            if (model.includes('insect')) {
                individualQuestionSet = DEAD_HEART_QUESTIONS;
            } else if (model.includes('disease')) {
                individualQuestionSet = DISEASE_QUESTIONS;
            }
            
            const container = document.getElementById('individualQuestionsContainer');
            container.innerHTML = '';
            
            for (let index = 0; index < Math.min(45, individualQuestionSet.length); index++) {
                createIndividualQuestionCard(individualQuestionSet[index], index, container);
            }
            
            resetIndividualQuiz();
        }

        function createIndividualQuestionCard(question, index, container) {
            const questionCard = document.createElement('div');
            questionCard.className = 'question-card';
            
            const illustration = getQuestionIllustration(question);
            
            questionCard.innerHTML = `
                <div class="question-header">
                    <div class="question-number">${index + 1}</div>
                    <div class="question-illustration">${illustration}</div>
                    <div class="question-text">${question}</div>
                </div>
                <div class="answer-options">
                    <button class="answer-btn" data-question="${index}" data-answer="1" onclick="selectIndividualAnswer(${index}, 1, this)">
                        ✅ Yes
                    </button>
                    <button class="answer-btn" data-question="${index}" data-answer="0" onclick="selectIndividualAnswer(${index}, 0, this)">
                        ❌ No
                    </button>
                </div>
            `;
            container.appendChild(questionCard);
        }

        function selectIndividualAnswer(questionIndex, answer, buttonElement) {
            const questionCard = buttonElement.closest('.question-card');
            questionCard.querySelectorAll('.answer-btn').forEach(btn => btn.classList.remove('selected'));
            
            buttonElement.classList.add('selected');
            individualAnswers[questionIndex] = answer;
            
            updateIndividualQuizProgress();
        }

        function updateIndividualQuizProgress() {
            const totalQuestions = 45;
            const answeredCount = Object.keys(individualAnswers).length;
            const progressPercent = Math.round((answeredCount / totalQuestions) * 100);
            
            document.getElementById('individualAnsweredCount').textContent = answeredCount;
            document.getElementById('individualProgressPercent').textContent = progressPercent + '%';
            document.getElementById('individualProgressBar').style.width = progressPercent + '%';
            
            document.getElementById('analyzeIndividualQuiz').disabled = answeredCount < totalQuestions;
        }

        function resetIndividualQuiz() {
            individualAnswers = {};
            document.querySelectorAll('#individualQuestionsContainer .answer-btn').forEach(btn => btn.classList.remove('selected'));
            updateIndividualQuizProgress();
        }

        function setupIndividualFileUpload() {
            const fileInput = document.getElementById('individualFileInput');
            
            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    handleIndividualFileSelect(file);
                }
            });
        }

        function setupIndividualDragDrop() {
            const uploadArea = document.getElementById('individualUploadArea');
            
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
                
                const file = e.dataTransfer.files[0];
                if (file) {
                    handleIndividualFileSelect(file);
                }
            });
        }

        function handleIndividualFileSelect(file) {
            individualFile = file;
            
            const preview = document.getElementById('individualFilePreview');
            const previewImage = document.getElementById('individualPreviewImage');
            const fileName = document.getElementById('individualFileName');
            
            fileName.textContent = file.name;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                preview.classList.add('show');
            };
            reader.readAsDataURL(file);
            
            document.getElementById('analyzeIndividualImage').disabled = false;
        }

        function analyzeIndividualImage() {
            if (!individualFile || !individualModel) return;

            const formData = new FormData();
            formData.append('file', individualFile);
            formData.append('model_type', individualModel);
            formData.append('input_type', 'image');
            formData.append('session_id', sessionId);

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
                    individualResult = data;
                    displayIndividualResults(data, 'image');
                }
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error.message);
            });
        }

        function analyzeIndividualQuiz() {
            if (!individualModel) return;

            const answersArray = [];
            for (let i = 0; i < 45; i++) {
                answersArray.push(individualAnswers[i] || 0);
            }

            const requestData = {
                model_type: individualModel,
                input_type: 'quiz',
                answers: answersArray,
                session_id: sessionId
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
                    individualResult = data;
                    displayIndividualResults(data, 'quiz');
                }
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error.message);
            });
        }

        function displayIndividualResults(result, analysisType) {
            document.getElementById('individualImageUpload').style.display = 'none';
            document.getElementById('individualQuiz').style.display = 'none';
            document.getElementById('individualResults').style.display = 'block';
            
            const modelType = individualModel.includes('disease') ? 'Disease' : 'Pest';
            const inputType = analysisType === 'image' ? 'Image' : 'Quiz';
            
            document.getElementById('individualResultsTitle').textContent = `${modelType} ${inputType} Analysis Results`;
            document.getElementById('individualResultsSubtitle').textContent = `${inputType}-based ${modelType.toLowerCase()} detection analysis`;
            document.getElementById('individualResultType').textContent = `${inputType} Analysis`;
            
            let confidence, resultText, recommendation;
            
            if (analysisType === 'image') {
                const detections = result.detections || [];
                confidence = detections.length > 0 ? Math.max(...detections.map(d => d.confidence || 0)) : 0;
                resultText = detections.length > 0 ? `${detections.length} Detection(s) Found` : 'No Detections Found';
                
                if (confidence > 0.7) {
                    recommendation = `High confidence ${modelType.toLowerCase()} detection. Immediate action recommended.`;
                } else if (confidence > 0.4) {
                    recommendation = `Moderate confidence detection. Monitor closely and consider preventive measures.`;
                } else {
                    recommendation = `No significant ${modelType.toLowerCase()} detected. Continue regular monitoring.`;
                }
            } else {
                const prediction = Array.isArray(result.prediction) ? result.prediction[0] : result.prediction;
                confidence = result.confidence || (prediction === 1 ? 0.8 : 0.2);
                resultText = prediction === 1 ? `${modelType} Detected` : `No ${modelType} Detected`;
                
                if (prediction === 1) {
                    recommendation = `Assessment indicates potential ${modelType.toLowerCase()} presence. Consider further inspection or expert consultation.`;
                } else {
                    recommendation = `Assessment suggests no immediate ${modelType.toLowerCase()} concerns. Continue regular monitoring.`;
                }
            }
            
            document.getElementById('individualConfidence').textContent = `${(confidence * 100).toFixed(1)}%`;
            document.getElementById('individualResult').textContent = resultText;
            
            setIndividualConfidenceColor(confidence);
            
            const detailedResults = document.getElementById('individualDetailedResults');
            detailedResults.innerHTML = `
                <div style="background: rgba(255,255,255,0.9); padding: 25px; border-radius: 15px; margin-top: 20px; border-left: 5px solid #4CAF50;">
                    <h4 style="color: #1B5E20; margin-bottom: 15px;">🎯 Recommendation:</h4>
                    <p style="font-size: 1.1em; line-height: 1.6; color: #333;">${recommendation}</p>
                    <div style="margin-top: 15px; padding: 15px; background: rgba(248,255,248,0.8); border-radius: 10px;">
                        <strong>Analysis Summary:</strong><br>
                        • Model: ${individualModel}<br>
                        • Input Type: ${inputType}<br>
                        • Confidence: ${(confidence * 100).toFixed(1)}%<br>
                        • Result: ${resultText}
                    </div>
                </div>
            `;
        }

        function setIndividualConfidenceColor(confidence) {
            const element = document.getElementById('individualConfidence');
            element.className = 'confidence-score';
            
            if (confidence > 0.7) {
                element.classList.add('high');
            } else if (confidence > 0.4) {
                element.classList.add('medium');
            } else {
                element.classList.add('low');
            }
        }

        function startNewIndividualAnalysis() {
            individualModel = null;
            individualInputType = null;
            individualAnswers = {};
            individualFile = null;
            individualQuestionSet = [];
            individualResult = null;
            
            document.querySelectorAll('#individualAnalysis .model-card').forEach(c => c.classList.remove('selected'));
            document.getElementById('analyzeIndividualImage').disabled = true;
            document.getElementById('analyzeIndividualQuiz').disabled = true;
            document.getElementById('individualFilePreview').classList.remove('show');
            
            backToMainSelector();
        }

        function downloadIndividualResults() {
            if (!individualResult) return;
            
            const report = {
                timestamp: new Date().toLocaleString(),
                session_id: sessionId,
                analysis_type: 'Individual Analysis',
                model: individualModel,
                input_type: individualInputType,
                result: individualResult,
                confidence: document.getElementById('individualConfidence').textContent,
                assessment: document.getElementById('individualResult').textContent
            };
            
            const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `individual_sugarcane_analysis_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        function backToMainSelector() {
            document.getElementById('mainSelector').classList.remove('hidden');
            document.getElementById('individualAnalysis').classList.add('hidden');
            document.getElementById('individualInterface').classList.remove('show');
            document.getElementById('fusionPipeline').classList.remove('show');
            
            document.querySelectorAll('.pipeline-card').forEach(c => c.classList.remove('selected'));
            currentPipeline = null;
        }

        function selectPipeline(pipeline) {
            currentPipeline = pipeline;
            
            document.querySelectorAll('.pipeline-card').forEach(c => c.classList.remove('selected'));
            document.querySelector(`[data-pipeline="${pipeline}"]`).classList.add('selected');
            
            if (pipeline === 'individual') {
                document.getElementById('mainSelector').classList.add('hidden');
                document.getElementById('individualAnalysis').classList.remove('hidden');
                document.getElementById('fusionPipeline').classList.remove('show');
            } else if (pipeline === 'fusion') {
                document.getElementById('mainSelector').classList.add('hidden');
                document.getElementById('individualAnalysis').classList.add('hidden');
                document.getElementById('fusionPipeline').classList.add('show');
            }
        }

        function selectTarget(target) {
            fusionTarget = target;
            
            document.querySelectorAll('#step1-content .pipeline-card').forEach(c => c.classList.remove('selected'));
            document.querySelector(`[data-target="${target}"]`).classList.add('selected');
            
            document.getElementById('proceedToQuiz').disabled = false;
        }

        function proceedToQuiz() {
            if (!fusionTarget) return;
            
            showStep(2);
            
            const quizTitle = fusionTarget === 'disease' ? 
                'Disease Assessment - Simple Questions' : 
                'Pest Assessment - Simple Questions';
            document.getElementById('quizStepTitle').textContent = `Step 2: ${quizTitle}`;
            
            generateQuiz();
        }

        function loadQuizQuestions() {
            fetch('/quiz_questions')
                .then(response => response.json())
                .then(data => {
                    DEAD_HEART_QUESTIONS = data.dead_heart_questions || [];
                    TILLER_QUESTIONS = data.tiller_questions || [];
                    DISEASE_QUESTIONS = data.disease_questions || [];
                })
                .catch(error => {
                    console.error('Error loading quiz questions:', error);
                });
        }

        function generateQuiz() {
            if (fusionTarget === 'esb') {
                currentQuestionSet = DEAD_HEART_QUESTIONS;
            } else if (fusionTarget === 'disease') {
                currentQuestionSet = DISEASE_QUESTIONS;
            }
            
            const container = document.getElementById('questionsContainer');
            container.innerHTML = '';
            
            for (let index = 0; index < Math.min(45, currentQuestionSet.length); index++) {
                createQuestionCard(currentQuestionSet[index], index, container);
            }
            
            resetQuiz();
        }

        function createQuestionCard(question, index, container) {
    const questionCard = document.createElement('div');
    questionCard.className = 'question-card';
    
    let illustration = getQuestionIllustration(question);
    if (!illustration) {
        illustration = `<div style="
            width:40px;
            height:40px;
            background-color:#eee;
            border-radius:4px;
            display:inline-block;
            flex-shrink:0;
        "></div>`;
    }

    questionCard.innerHTML = `
        <div class="question-header" style="display:flex; align-items:center; justify-content:space-between; gap:10px;">
            <!-- Left side: number, illustration, text -->
            <div style="display:flex; align-items:center; gap:10px; flex:1;">
                <div class="question-number">${index + 1}</div>
                <div class="question-illustration">${illustration}</div>
                <div class="question-text">${question}</div>
            </div>
            <!-- Right side: fixed gray placeholder -->
            <div class="question-image-placeholder" style="
                width:150px;
                height:150px;
                background-color:#ccc;
                background-image:url('/static/placeholder.jpg');
                background-size:cover;
                border-radius:4px;
                flex-shrink:0;
            "></div>
        </div>
        <div class="answer-options" style="margin-top:8px;">
            <button class="answer-btn" data-question="${index}" data-answer="1" onclick="selectAnswer(${index}, 1, this)">
                ✅ Yes
            </button>
            <button class="answer-btn" data-question="${index}" data-answer="0" onclick="selectAnswer(${index}, 0, this)">
                ❌ No
            </button>
        </div>
    `;

    container.appendChild(questionCard);
}


        function selectAnswer(questionIndex, answer, buttonElement) {
            const questionCard = buttonElement.closest('.question-card');
            questionCard.querySelectorAll('.answer-btn').forEach(btn => btn.classList.remove('selected'));
            
            buttonElement.classList.add('selected');
            quizAnswers[questionIndex] = answer;
            
            updateQuizProgress();
        }

        function updateQuizProgress() {
            const totalQuestions = 45;
            const answeredCount = Object.keys(quizAnswers).length;
            const progressPercent = Math.round((answeredCount / totalQuestions) * 100);
            
            document.getElementById('answeredCount').textContent = answeredCount;
            document.getElementById('progressPercent').textContent = progressPercent + '%';
            document.getElementById('progressBar').style.width = progressPercent + '%';
            
            document.getElementById('completeQuiz').disabled = answeredCount < totalQuestions;
        }

        function resetQuiz() {
            quizAnswers = {};
            document.querySelectorAll('#questionsContainer .answer-btn').forEach(btn => btn.classList.remove('selected'));
            updateQuizProgress();
        }

        function completeQuiz() {
            const answersArray = [];
            for (let i = 0; i < 45; i++) {
                answersArray.push(quizAnswers[i] || 0);
            }

            const requestData = {
                model_type: fusionTarget === 'disease' ? 'disease_tabnet' : 'insect_tabnet',
                input_type: 'quiz',
                answers: answersArray,
                session_id: sessionId
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
                    quizResult = data;
                    showStep(3);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error.message);
            });
        }

        function setupFileUpload() {
            const fileInput = document.getElementById('fusionFileInput');
            
            fileInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    handleFileSelect(file);
                }
            });
        }

        function setupDragDrop() {
            const uploadArea = document.getElementById('imageUploadArea');
            
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                this.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                this.classList.remove('dragover');
                
                const file = e.dataTransfer.files[0];
                if (file) {
                    handleFileSelect(file);
                }
            });
        }

        function handleFileSelect(file) {
            selectedFile = file;
            
            const preview = document.getElementById('filePreview');
            const previewImage = document.getElementById('previewImage');
            const fileName = document.getElementById('fileName');
            
            fileName.textContent = file.name;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                preview.classList.add('show');
            };
            reader.readAsDataURL(file);
            
            document.getElementById('startFusionAnalysis').disabled = false;
        }

        function startFusionAnalysis() {
            if (!selectedFile || !quizResult) return;

            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('model_type', fusionTarget === 'disease' ? 'disease_yolo' : 'insect_yolo');
            formData.append('input_type', 'image');
            formData.append('session_id', sessionId);

            showLoading();
            showStep(4);

            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    hideLoading();
                    showError(data.error);
                } else {
                    imageResult = data;
                    return fetch('/fusion_analysis', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            session_id: sessionId
                        })
                    });
                }
            })
            .then(response => {
                if (response) {
                    return response.json();
                }
            })
            .then(fusionData => {
                hideLoading();
                if (fusionData && fusionData.error) {
                    showError(fusionData.error);
                } else if (fusionData) {
                    displayFusionResults(fusionData);
                }
            })
            .catch(error => {
                hideLoading();
                showError('Network error: ' + error.message);
            });
        }

        function displayFusionResults(fusionData) {
            const quizResult = fusionData.quiz_result;
            const imageResult = fusionData.image_result;
            const fusionAnalysis = fusionData.fusion_analysis;

            const quizPrediction = Array.isArray(quizResult.prediction) ? quizResult.prediction[0] : quizResult.prediction;
            const quizConfidence = fusionAnalysis.quiz_confidence;
            const imageConfidence = fusionAnalysis.image_confidence;
            const fusionScore = fusionAnalysis.fusion_score;
            const detectionCount = fusionAnalysis.detection_count;

            document.getElementById('quizConfidence').textContent = `${(quizConfidence * 100).toFixed(1)}%`;
            document.getElementById('imageConfidence').textContent = `${(imageConfidence * 100).toFixed(1)}%`;
            document.getElementById('fusionConfidence').textContent = `${(fusionScore * 100).toFixed(1)}%`;
            
            setConfidenceColor('quizConfidence', quizConfidence);
            setConfidenceColor('imageConfidence', imageConfidence);
            setConfidenceColor('fusionConfidence', fusionScore);
            
            const targetName = fusionTarget === 'disease' ? 'Disease' : 'Pest Problem';
            const quizResultText = quizPrediction === 1 ? `${targetName} Detected` : `No ${targetName} Detected`;
            const imageResultText = detectionCount > 0 ? `${detectionCount} Detection(s) Found` : 'No Detections Found';
            
            let fusionResultText = '✅ No Issues Detected';
            let recommendation = 'Your crop looks healthy. Continue regular monitoring.';
            
            if (fusionScore > 0.7) {
                fusionResultText = `⚠️ High Risk: ${targetName} Likely Present`;
                recommendation = `Immediate action recommended. Consult agricultural expert or apply appropriate treatment.`;
            } else if (fusionScore > 0.4) {
                fusionResultText = `⚡ Moderate Risk: Possible ${targetName}`;
                recommendation = `Monitor closely and consider preventive measures. Early intervention may be needed.`;
            }
            
            document.getElementById('quizResult').textContent = quizResultText;
            document.getElementById('imageResult').textContent = imageResultText;
            document.getElementById('fusionResult').textContent = fusionResultText;
            
            const detailedResults = document.getElementById('detailedResults');
            detailedResults.innerHTML = `
                <div style="background: rgba(255,255,255,0.9); padding: 25px; border-radius: 15px; margin-top: 20px; border-left: 5px solid #4CAF50;">
                    <h4 style="color: #1B5E20; margin-bottom: 15px;">🎯 Recommendation:</h4>
                    <p style="font-size: 1.1em; line-height: 1.6; color: #333;">${recommendation}</p>
                    <div style="margin-top: 15px; padding: 15px; background: rgba(248,255,248,0.8); border-radius: 10px;">
                        <strong>Analysis Summary:</strong><br>
                        • Quiz Assessment: ${(quizConfidence * 100).toFixed(1)}% confidence<br>
                        • Image Analysis: ${detectionCount} detections with ${(imageConfidence * 100).toFixed(1)}% confidence<br>
                        • Combined AI Score: ${(fusionScore * 100).toFixed(1)}% risk level
                    </div>
                </div>
            `;
        }

        function setConfidenceColor(elementId, confidence) {
            const element = document.getElementById(elementId);
            element.className = 'confidence-score';
            
            if (confidence > 0.7) {
                element.classList.add('high');
            } else if (confidence > 0.4) {
                element.classList.add('medium');
            } else {
                element.classList.add('low');
            }
        }

        function showStep(stepNumber) {
            document.querySelectorAll('.step-content').forEach(content => {
                content.classList.remove('active');
            });
            
            document.getElementById(`step${stepNumber}-content`).classList.add('active');
            currentStep = stepNumber;
        }

        function goBackToTarget() {
            showStep(1);
        }

        function goBackToQuiz() {
            showStep(2);
        }

        function startNewAnalysis() {
            if (sessionId) {
                fetch(`/clear_session/${sessionId}`)
                    .catch(error => console.log('Session clear error:', error));
            }
            
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            
            fusionTarget = null;
            quizAnswers = {};
            selectedFile = null;
            quizResult = null;
            imageResult = null;
            
            document.querySelectorAll('.pipeline-card').forEach(c => c.classList.remove('selected'));
            document.getElementById('proceedToQuiz').disabled = true;
            document.getElementById('completeQuiz').disabled = true;
            document.getElementById('startFusionAnalysis').disabled = true;
            document.getElementById('filePreview').classList.remove('show');
            
            showStep(1);
        }

        function downloadResults() {
            const report = {
                timestamp: new Date().toLocaleString(),
                session_id: sessionId,
                detection_target: fusionTarget === 'disease' ? 'Disease Detection' : 'Pest Detection',
                simple_quiz_format: '45 farmer-friendly questions with visual illustrations',
                quiz_result: {
                    confidence: document.getElementById('quizConfidence').textContent,
                    result: document.getElementById('quizResult').textContent
                },
                image_result: {
                    confidence: document.getElementById('imageConfidence').textContent,
                    result: document.getElementById('imageResult').textContent
                },
                smart_ai_fusion: {
                    overall_score: document.getElementById('fusionConfidence').textContent,
                    final_assessment: document.getElementById('fusionResult').textContent
                }
            };
            
            const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `simple_sugarcane_analysis_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
        }

        function showLoading() {
            document.getElementById('loading').style.display = 'block';
        }

        function hideLoading() {
            document.getElementById('loading').style.display = 'none';
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
            
            setTimeout(() => {
                errorDiv.style.display = 'none';
            }, 5000);
        }

        // Keyboard navigation support
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                if (document.getElementById('individualInterface').classList.contains('show')) {
                    backToMainSelector();
                } else if (document.getElementById('fusionPipeline').classList.contains('show')) {
                    if (currentStep > 1) {
                        if (currentStep === 2) goBackToTarget();
                        else if (currentStep === 3) goBackToQuiz();
                    } else {
                        backToMainSelector();
                    }
                }
            }
        });

        console.log('🌾 Simple Sugarcane AI - Complete System Ready!');
        console.log('Features: Individual Analysis, Fusion Pipeline, 45-Question Assessment, Image Analysis');