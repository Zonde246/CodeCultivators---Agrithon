from flask import Flask, request, render_template, jsonify, redirect, url_for
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import os
import zipfile
import pickle
import io
import base64
from werkzeug.utils import secure_filename
import json
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store models
yolo_disease_model = None  # YOLOv8s-seg model
yolo_insect_model = None   # YOLOv8s model
tabnet_disease_model = None
tabnet_insect_model = None

# YOLO class names (you may need to update these based on your specific models)
DISEASE_CLASSES = [
    'healthy', 'bacterial_blight', 'brown_spot', 'leaf_blast', 
    'tungro', 'hispa', 'downy_mildew', 'bacterial_leaf_streak'
]

INSECT_CLASSES = [
    'armyworm', 'beetle', 'bollworm', 'grasshopper', 
    'mites', 'sawfly', 'stem_borer', 'thrips'
]

# Quiz questions for reference (these match the frontend)
ARMYWORM_QUESTIONS = [
    "Is the pest in the image an armyworm?",
    "Is the armyworm green in color?",
    "Is the armyworm brown in color?",
    "Is the armyworm found on the leaf top?",
    "Is the armyworm found on the underside of the leaf?",
    "Is the armyworm present on the stem?",
    "Is the armyworm feeding on the crop?",
    "Are visible bite marks present on the leaf?",
    "Are there multiple armyworms in the image?",
    "Is any frass (armyworm waste) visible near the pest?",
    "Are eggs visible near the armyworm?",
    "Are larvae of the armyworm visible?",
    "Has the crop been attacked by armyworm in previous seasons?",
    "Was pesticide recently applied to this crop area?",
    "Is the armyworm population increasing?",
    "Is the armyworm active during daylight hours?",
    "Is the armyworm mostly active during night?",
    "Is the leaf portion of the plant affected?",
    "Is the stem portion of the plant affected?",
    "Is the damage restricted to a small part of the crop?",
    "Are nearby plants also showing signs of armyworm infestation?",
    "Is the armyworm moving actively?",
    "Are there signs of curled leaves due to feeding?",
    "Has the armyworm damaged more than one section of the same plant?",
    "Is there visible discoloration of the crop due to pest feeding?",
    "Does the armyworm show striping or lines on its body?",
    "Is the length of the armyworm greater than 20 mm?",
    "Are any dead armyworms seen in the area (possibly due to pesticide)?",
    "Is any chewing sound audible during the inspection?",
    "Has any farmer nearby reported armyworm infestation in the last week?"
]

DISEASE_QUESTIONS = [
    "Is there a yellow halo around the spots?",
    "Are the leaf spots circular with concentric rings?",
    "Does the disease begin on the lower leaves?",
    "Are the lesions expanding over time?",
    "Is the center of the spot dry and brown?",
    "Are multiple spots merging to form large blotches?",
    "Does the leaf show signs of early yellowing?",
    "Are stems or fruits also affected?",
    "Are the affected leaves wilting?",
    "Is the infection spreading upward on the plant?",
    "Are concentric rings visible clearly on the leaves?",
    "Is there any rotting seen on fruit?",
    "Are the leaf margins turning brown?",
    "Is the plant under moisture stress?",
    "Is the disease more active during rainy days?",
    "Are nearby tomato plants also showing similar symptoms?",
    "Is there any black moldy growth on the lesion?",
    "Does the disease affect the whole plant?",
    "Is the spot size more than 5mm in diameter?",
    "Are the lesions visible on both sides of the leaf?",
    "Is the infection found only on mature leaves?",
    "Are the leaf veins visible through the lesion?",
    "Is the damage uniform across the field?",
    "Was there previous history of Early Blight in this field?",
    "Is the farmer using resistant tomato varieties?",
    "Was any fungicide recently applied?",
    "Was there poor air circulation in the field?",
    "Was the field irrigated from overhead sprinklers?",
    "Are pruning and sanitation practices followed?",
    "Is there any other crop in the field showing similar spots?"
]

class ModelLoader:
    @staticmethod
    def load_yolo_model(model_path):
        """Load YOLO model using ultralytics"""
        try:
            model = YOLO(model_path)
            return model
        except Exception as e:
            print(f"Error loading YOLO model {model_path}: {e}")
            return None
    
    @staticmethod
    def load_tabnet_model(model_path):
        """Load TabNet model from zip file using pytorch-tabnet"""
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
            
            print(f"Loading TabNet model from: {model_path}")
            
            # Try loading as classifier first
            try:
                clf = TabNetClassifier()
                clf.load_model(model_path)
                print(f"Successfully loaded TabNet classifier from {model_path}")
                return clf
            except Exception as e:
                print(f"Failed to load as classifier: {e}")
                
                # Try loading as regressor
                try:
                    regressor = TabNetRegressor()
                    regressor.load_model(model_path)
                    print(f"Successfully loaded TabNet regressor from {model_path}")
                    return regressor
                except Exception as e2:
                    print(f"Failed to load as regressor: {e2}")
                    return None
            
        except ImportError:
            print("pytorch_tabnet not installed. Please install it with: pip install pytorch-tabnet")
            return None
        except Exception as e:
            print(f"Error loading TabNet model {model_path}: {e}")
            return None

def load_all_models():
    """Load all models at startup"""
    global yolo_disease_model, yolo_insect_model, tabnet_disease_model, tabnet_insect_model
    
    # Load YOLO models
    if os.path.exists('diseaseBest.pt'):
        yolo_disease_model = ModelLoader.load_yolo_model('diseaseBest.pt')
        print("Disease YOLO model (YOLOv8s-seg) loaded successfully")
    
    if os.path.exists('insectsBest.pt'):
        yolo_insect_model = ModelLoader.load_yolo_model('insectsBest.pt')
        print("Insect YOLO model (YOLOv8s) loaded successfully")
    
    # Load TabNet models
    if os.path.exists('tabnet_disease_model.zip'):
        tabnet_disease_model = ModelLoader.load_tabnet_model('tabnet_disease_model.zip')
        if tabnet_disease_model is not None:
            print("Disease TabNet model loaded successfully")
        else:
            print("Failed to load Disease TabNet model")
    
    if os.path.exists('tabnet_armyworm_detector.zip'):
        tabnet_insect_model = ModelLoader.load_tabnet_model('tabnet_armyworm_detector.zip')
        if tabnet_insect_model is not None:
            print("Insect TabNet model loaded successfully")
        else:
            print("Failed to load Insect TabNet model")

def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'csv', 'xlsx', 'xls'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def draw_segmentation_results(image, results, class_names):
    """Draw segmentation results on image"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Create a copy for drawing
        result_img = img_array.copy()
        
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Color map for different classes
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
            
            for i, (mask, box, conf, cls) in enumerate(zip(masks, boxes, confidences, classes)):
                # Resize mask to image dimensions
                mask_resized = cv2.resize(mask, (img_array.shape[1], img_array.shape[0]))
                
                # Apply colored mask
                color = colors[cls][:3]  # RGB values
                colored_mask = (mask_resized[..., None] * np.array(color) * 255).astype(np.uint8)
                
                # Blend mask with original image
                result_img = cv2.addWeighted(result_img, 0.7, colored_mask, 0.3, 0)
                
                # Draw bounding box
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(result_img, (x1, y1), (x2, y2), 
                            (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)
                
                # Add label
                label = f'{class_names[cls]}: {conf:.2f}'
                cv2.putText(result_img, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)
        
        return Image.fromarray(result_img)
    except Exception as e:
        print(f"Error drawing segmentation results: {e}")
        return image

def draw_detection_results(image, results, class_names):
    """Draw detection results on image"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        result_img = img_array.copy()
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            # Color map for different classes
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box.astype(int)
                color = colors[cls][:3]  # RGB values
                
                # Draw bounding box
                cv2.rectangle(result_img, (x1, y1), (x2, y2), 
                            (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)
                
                # Add label
                label = f'{class_names[cls]}: {conf:.2f}'
                cv2.putText(result_img, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (int(color[0]*255), int(color[1]*255), int(color[2]*255)), 2)
        
        return Image.fromarray(result_img)
    except Exception as e:
        print(f"Error drawing detection results: {e}")
        return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def predict_yolo_image(model, image, class_names, is_segmentation=False):
    """Make prediction using YOLO model on image"""
    try:
        # Run inference
        results = model(image)
        
        # Extract predictions
        predictions = []
        confidences = []
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, cls in zip(boxes, confs, classes):
                predictions.append({
                    'class': class_names[cls] if cls < len(class_names) else f'class_{cls}',
                    'confidence': float(conf),
                    'bbox': box.tolist()
                })
                confidences.append(float(conf))
        
        # Draw results on image
        if is_segmentation:
            result_image = draw_segmentation_results(image, results, class_names)
        else:
            result_image = draw_detection_results(image, results, class_names)
        
        # Convert result image to base64
        result_image_b64 = image_to_base64(result_image)
        
        return predictions, confidences, result_image_b64
        
    except Exception as e:
        print(f"Error in YOLO prediction: {e}")
        return None, None, None

def predict_tabnet_quiz(model, answers):
    """Make prediction using TabNet model on quiz answers"""
    try:
        print(f"TabNet quiz prediction input: {answers}")
        print(f"TabNet model type: {type(model)}")
        
        # Convert answers list to numpy array with correct shape
        data_array = np.array(answers, dtype=np.float32).reshape(1, -1)  # Shape: (1, 30)
        
        print(f"Data array shape for prediction: {data_array.shape}")
        print(f"Data array dtype: {data_array.dtype}")
        
        # Ensure exactly 30 features
        if data_array.shape[1] != 30:
            raise ValueError(f"Expected 30 features, got {data_array.shape[1]} features.")
        
        # Make prediction using TabNet model
        prediction = model.predict(data_array)
        print(f"Prediction result: {prediction}")
        print(f"Prediction shape: {prediction.shape}")
        
        # Try to get prediction probabilities if available
        probabilities = None
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data_array)
                print(f"Probabilities shape: {probabilities.shape}")
                print(f"Probabilities: {probabilities}")
        except Exception as e:
            print(f"Could not get probabilities: {e}")
        
        return prediction, probabilities
        
    except Exception as e:
        print(f"Error in TabNet quiz prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_tabnet_data(model, data):
    """Make prediction using TabNet model on structured data (for CSV uploads)"""
    try:
        print(f"TabNet prediction input data shape: {data.shape}")
        print(f"TabNet model type: {type(model)}")
        
        # Convert DataFrame to numpy array if needed
        if isinstance(data, pd.DataFrame):
            # Fill any remaining NaN values with 0
            data_clean = data.fillna(0)
            data_array = data_clean.values.astype(np.float32)
        else:
            data_array = np.array(data, dtype=np.float32)
        
        print(f"Data array shape for prediction: {data_array.shape}")
        print(f"Data array dtype: {data_array.dtype}")
        
        # Final safety check - ensure exactly 30 features
        if data_array.shape[1] != 30:
            if data_array.shape[1] > 30:
                print(f"Warning: Truncating from {data_array.shape[1]} to 30 features")
                data_array = data_array[:, :30]
            else:
                raise ValueError(f"Expected 30 features, got {data_array.shape[1]} features. "
                               f"Please ensure your data has exactly 30 feature columns (Q1-Q30).")
        
        print(f"Final data array shape: {data_array.shape}")
        
        # Make prediction using TabNet model
        prediction = model.predict(data_array)
        print(f"Prediction result: {prediction}")
        print(f"Prediction shape: {prediction.shape}")
        
        # Try to get prediction probabilities if available
        probabilities = None
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data_array)
                print(f"Probabilities shape: {probabilities.shape}")
        except Exception as e:
            print(f"Could not get probabilities: {e}")
        
        return prediction, probabilities
        
    except Exception as e:
        print(f"Error in TabNet prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, None

@app.route('/')
def index():
    """Main page with all options"""
    return render_template('index.html')

@app.route('/predict_quiz', methods=['POST'])
def predict_quiz():
    """Handle quiz-based predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        model_type = data.get('model_type')
        input_type = data.get('input_type')
        answers = data.get('answers')
        
        if not model_type or input_type != 'quiz' or not answers:
            return jsonify({'error': 'Invalid request data'}), 400
        
        if len(answers) != 30:
            return jsonify({'error': f'Expected 30 answers, got {len(answers)}'}), 400
        
        # Validate answers are binary (0 or 1)
        if not all(answer in [0, 1] for answer in answers):
            return jsonify({'error': 'All answers must be 0 (No) or 1 (Yes)'}), 400
        
        # Select appropriate model
        if model_type == 'disease_tabnet':
            if tabnet_disease_model is None:
                return jsonify({'error': 'Disease TabNet model not loaded'}), 500
            prediction, probabilities = predict_tabnet_quiz(tabnet_disease_model, answers)
        elif model_type == 'insect_tabnet':
            if tabnet_insect_model is None:
                return jsonify({'error': 'Insect TabNet model not loaded'}), 500
            prediction, probabilities = predict_tabnet_quiz(tabnet_insect_model, answers)
        else:
            return jsonify({'error': 'Invalid model type for quiz input'}), 400
        
        if prediction is not None:
            result = {
                'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                'model_used': model_type,
                'input_type': 'quiz',
                'num_questions_answered': len(answers)
            }
            if probabilities is not None:
                result['probabilities'] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Quiz prediction failed - check server logs for details'}), 500
            
    except Exception as e:
        print(f"Error in quiz prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle predictions based on model type and input"""
    try:
        model_type = request.form.get('model_type')
        input_type = request.form.get('input_type')
        
        if not model_type or not input_type:
            return jsonify({'error': 'Model type and input type are required'}), 400
        
        # Handle image input
        if input_type == 'image':
            if 'file' not in request.files:
                return jsonify({'error': 'No file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type'}), 400
            
            # Load and process image
            image = Image.open(file.stream)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Select appropriate model
            if model_type == 'disease_yolo':
                if yolo_disease_model is None:
                    return jsonify({'error': 'Disease YOLO model not loaded'}), 500
                predictions, confidences, result_image = predict_yolo_image(
                    yolo_disease_model, image, DISEASE_CLASSES, is_segmentation=True)
            elif model_type == 'insect_yolo':
                if yolo_insect_model is None:
                    return jsonify({'error': 'Insect YOLO model not loaded'}), 500
                predictions, confidences, result_image = predict_yolo_image(
                    yolo_insect_model, image, INSECT_CLASSES, is_segmentation=False)
            else:
                return jsonify({'error': 'Invalid model type for image input'}), 400
            
            if predictions is not None:
                return jsonify({
                    'predictions': predictions,
                    'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                    'detection_count': len(predictions),
                    'result_image': result_image,
                    'model_used': model_type
                })
            else:
                return jsonify({'error': 'Prediction failed'}), 500
        
        # Handle structured data input (CSV/Excel files)
        elif input_type == 'data':
            if 'file' not in request.files:
                return jsonify({'error': 'No data file uploaded'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Read structured data
            try:
                if file.filename.endswith('.csv'):
                    data = pd.read_csv(file.stream)
                elif file.filename.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(file.stream)
                else:
                    return jsonify({'error': 'Unsupported data file format'}), 400
                
                print(f"Data loaded: shape {data.shape}")
                print(f"Columns: {list(data.columns)}")
                print(f"First few rows:\n{data.head()}")
                
            except Exception as e:
                return jsonify({'error': f'Error reading data file: {str(e)}'}), 400
            
            # Basic data preprocessing
            try:
                print(f"Original data shape: {data.shape}")
                print(f"Original columns: {list(data.columns)}")
                
                # Remove any unnamed columns
                data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
                
                # Remove target column if present (common names: 'Target', 'target', 'label', 'Label', 'y')
                target_columns = ['Target', 'target', 'label', 'Label', 'y', 'class', 'Class']
                for col in target_columns:
                    if col in data.columns:
                        print(f"Removing target column: {col}")
                        data = data.drop(columns=[col])
                
                # If we still have more than 30 columns, keep only Q1-Q30
                if len(data.columns) > 30:
                    q_columns = [f'Q{i}' for i in range(1, 31)]
                    available_q_columns = [col for col in q_columns if col in data.columns]
                    if len(available_q_columns) == 30:
                        print("Using Q1-Q30 columns for prediction")
                        data = data[available_q_columns]
                    else:
                        # If Q columns don't exist, use first 30 columns
                        print(f"Q1-Q30 columns not found, using first 30 columns")
                        data = data.iloc[:, :30]
                
                print(f"Data shape after column selection: {data.shape}")
                print(f"Final columns: {list(data.columns)}")
                
                # Handle missing values
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                categorical_columns = data.select_dtypes(include=['object']).columns
                
                # Fill numeric NaN with median or 0
                for col in numeric_columns:
                    if data[col].isnull().any():
                        median_val = data[col].median()
                        if pd.isna(median_val):
                            data[col] = data[col].fillna(0)
                        else:
                            data[col] = data[col].fillna(median_val)
                
                # Fill categorical NaN with mode or 'unknown'
                for col in categorical_columns:
                    if data[col].isnull().any():
                        if data[col].mode().empty:
                            data[col] = data[col].fillna('unknown')
                        else:
                            data[col] = data[col].fillna(data[col].mode()[0])
                
                # Convert categorical to numeric using label encoding
                if len(categorical_columns) > 0:
                    from sklearn.preprocessing import LabelEncoder
                    label_encoders = {}
                    for col in categorical_columns:
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col].astype(str))
                        label_encoders[col] = le
                
                # Ensure we have exactly 30 features for TabNet
                if data.shape[1] != 30:
                    if data.shape[1] < 30:
                        # Pad with zeros if we have fewer than 30 features
                        missing_cols = 30 - data.shape[1]
                        for i in range(missing_cols):
                            data[f'feature_{data.shape[1] + i + 1}'] = 0
                        print(f"Padded data with {missing_cols} zero columns")
                    else:
                        # Truncate if we have more than 30 features
                        data = data.iloc[:, :30]
                        print(f"Truncated data to 30 features")
                
                print(f"Final preprocessed data shape: {data.shape}")
                print(f"Data types:\n{data.dtypes}")
                
                # Verify we have exactly 30 columns
                if data.shape[1] != 30:
                    raise ValueError(f"Expected 30 features, got {data.shape[1]} features")
                
            except Exception as e:
                print(f"Warning: Data preprocessing failed: {e}")
                # If preprocessing fails, try to use the data as-is but remove target column
                if 'Target' in data.columns:
                    data = data.drop(columns=['Target'])
                if data.shape[1] > 30:
                    data = data.iloc[:, :30]
            
            # Select appropriate model
            if model_type == 'disease_tabnet':
                if tabnet_disease_model is None:
                    return jsonify({'error': 'Disease TabNet model not loaded'}), 500
                prediction, probabilities = predict_tabnet_data(tabnet_disease_model, data)
            elif model_type == 'insect_tabnet':
                if tabnet_insect_model is None:
                    return jsonify({'error': 'Insect TabNet model not loaded'}), 500
                prediction, probabilities = predict_tabnet_data(tabnet_insect_model, data)
            else:
                return jsonify({'error': 'Invalid model type for data input'}), 400
            
            if prediction is not None:
                result = {
                    'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
                    'model_used': model_type,
                    'data_shape': data.shape,
                    'num_samples': len(data),
                    'input_type': 'data'
                }
                if probabilities is not None:
                    result['probabilities'] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
                
                return jsonify(result)
            else:
                return jsonify({'error': 'Prediction failed - check server logs for details'}), 500
        
        else:
            return jsonify({'error': 'Invalid input type'}), 400
            
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/model_status')
def model_status():
    """Check which models are loaded"""
    status = {
        'disease_yolo': yolo_disease_model is not None,
        'insect_yolo': yolo_insect_model is not None,
        'disease_tabnet': tabnet_disease_model is not None,
        'insect_tabnet': tabnet_insect_model is not None
    }
    return jsonify(status)

@app.route('/quiz_questions')
def get_quiz_questions():
    """Get quiz questions for frontend"""
    return jsonify({
        'armyworm_questions': ARMYWORM_QUESTIONS,
        'disease_questions': DISEASE_QUESTIONS
    })

@app.route('/class_names')
def get_class_names():
    """Get class names for YOLO models"""
    return jsonify({
        'disease_classes': DISEASE_CLASSES,
        'insect_classes': INSECT_CLASSES
    })

@app.route('/debug_tabnet')
def debug_tabnet():
    """Debug endpoint for TabNet models"""
    debug_info = {
        'disease_model_loaded': tabnet_disease_model is not None,
        'insect_model_loaded': tabnet_insect_model is not None,
        'disease_model_type': str(type(tabnet_disease_model)) if tabnet_disease_model else None,
        'insect_model_type': str(type(tabnet_insect_model)) if tabnet_insect_model else None,
        'pytorch_tabnet_available': False
    }
    
    # Check if pytorch-tabnet is available
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier
        debug_info['pytorch_tabnet_available'] = True
        debug_info['tabnet_version'] = 'Available'
    except ImportError:
        debug_info['pytorch_tabnet_version'] = 'Not installed'
    
    # Add model methods info
    if tabnet_disease_model:
        debug_info['disease_model_methods'] = [method for method in dir(tabnet_disease_model) 
                                               if not method.startswith('_')]
    
    if tabnet_insect_model:
        debug_info['insect_model_methods'] = [method for method in dir(tabnet_insect_model) 
                                              if not method.startswith('_')]
    
    return jsonify(debug_info)

@app.route('/test_quiz')
def test_quiz():
    """Test endpoint for quiz functionality"""
    # Test data: all "Yes" answers for armyworm detection
    test_answers = [1] * 30
    
    if tabnet_insect_model is None:
        return jsonify({'error': 'Insect TabNet model not loaded'})
    
    try:
        prediction, probabilities = predict_tabnet_quiz(tabnet_insect_model, test_answers)
        
        result = {
            'test_input': test_answers,
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'success': True
        }
        
        if probabilities is not None:
            result['probabilities'] = probabilities.tolist() if hasattr(probabilities, 'tolist') else probabilities
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == '__main__':
    # Load models at startup
    print("Loading models...")
    load_all_models()
    
    # Start Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)