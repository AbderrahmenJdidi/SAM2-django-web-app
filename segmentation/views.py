import os
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .forms import ImageUploadForm
from .models import ImageUpload
from datetime import datetime
from PIL import Image
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import MobileViTForImageClassification
import logging
from rest_framework.decorators import api_view
import io
import cv2
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = None
cnn_classifier = None

class AnesthesiaClassifier:
    def __init__(self, model_path, optimal_threshold=0.5379):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Thresholds from your training
        self.thresholds = {
            'conservative': max(0.35, optimal_threshold - 0.10),
            'balanced': max(0.45, optimal_threshold - 0.05),
            'precise': optimal_threshold
        }
        
        # Transform - same as your validation transform
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.model = self._load_model()
        logger.info(f"CNN Classifier initialized with thresholds: {self.thresholds}")
    
    def _load_model(self):
        """Load the trained model"""
        try:
            model = MobileViTForImageClassification.from_pretrained(
                'apple/mobilevit-xx-small',
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            model.classifier = nn.Sequential(
                nn.Dropout(0.8),
                nn.Linear(model.classifier.in_features, 2)
            )
            
            # Load your trained weights
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            logger.info(f"Model loaded successfully on {self.device}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def predict(self, image_path, threshold_mode='balanced'):
        """Predict on segmented image"""
        if self.model is None:
            return {'error': 'Model not loaded properly'}
        
        threshold = self.thresholds[threshold_mode]
        
        try:
            # Load and preprocess image
            img = Image.open(image_path).convert('RGB')
            tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(tensor).logits
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                
                prob_safe = probs[0, 1]
                prob_hard = probs[0, 0]
                
                is_hard = prob_hard >= threshold
                prediction = 'hard' if is_hard else 'safe'
                confidence = max(prob_safe, prob_hard)
                
                return {
                    'prediction': prediction,
                    'probabilities': {'safe': float(prob_safe), 'hard': float(prob_hard)},
                    'confidence': float(confidence),
                    'threshold_used': threshold,
                    'threshold_mode': threshold_mode,
                    'clinical_recommendation': self._get_clinical_recommendation(prob_hard, confidence),
                    'risk_level': self._assess_risk_level(prob_hard)
                }
        
        except Exception as e:
            logger.error(f"CNN prediction error: {e}")
            return {'error': str(e)}
    
    def _get_clinical_recommendation(self, prob_hard, confidence):
        """Clinical recommendations based on probability and confidence"""
        if prob_hard >= 0.75:
            return "HIGH RISK: Prepare difficult intubation equipment, experienced anesthesiologist required"
        elif prob_hard >= 0.60:
            return "MODERATE RISK: Enhanced preparation recommended, increased monitoring"
        elif prob_hard >= 0.45:
            return "ATTENTION: Additional clinical evaluation recommended"
        elif confidence < 0.65:
            return "UNCERTAINTY: Low confidence - Thorough physical examination necessary"
        else:
            return "STANDARD RISK: Standard anesthetic procedure, normal monitoring"
    
    def _assess_risk_level(self, prob_hard):
        """Assess risk level"""
        if prob_hard >= 0.70:
            return "HIGH"
        elif prob_hard >= 0.55:
            return "MODERATE"
        elif prob_hard >= 0.40:
            return "LOW-MODERATE"
        else:
            return "LOW"

def load_sam2_model():
    global predictor
    if predictor is None:
        checkpoint = "C:\\Users\\jdidi\\OneDrive\\Bureau\\stage_2eme\\application\\sam2_django_app\\segment-anything-2\\checkpoints\\sam2.1_hiera_tiny.pt"
        model_cfg = "C:\\Users\\jdidi\\OneDrive\\Bureau\\stage_2eme\\application\\sam2_django_app\\segment-anything-2\\sam2\\configs\\sam2.1\\sam2.1_hiera_t.yaml"
        device = torch.device("cpu")
        sam2_model = build_sam2(model_cfg, checkpoint, device)
        predictor = SAM2ImagePredictor(sam2_model)
    return predictor

def load_cnn_classifier():
    global cnn_classifier
    if cnn_classifier is None:
        # Update this path to your actual model location
        MODEL_PATH = "C:\\Users\\jdidi\\OneDrive\\Bureau\\stage_2eme\\application\\sam2_django_app\\models\\best_anesthesia_model (5).pth"
        OPTIMAL_THRESHOLD = 0.5379
        
        if os.path.exists(MODEL_PATH):
            cnn_classifier = AnesthesiaClassifier(MODEL_PATH, OPTIMAL_THRESHOLD)
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            cnn_classifier = None
    return cnn_classifier

def detect_eye_center(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    """Detect the center of the first eye in the image using OpenCV Haar cascades."""
    # Load Haar cascade classifiers for face and eyes
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    if face_cascade.empty() or eye_cascade.empty():
        logger.error("Failed to load Haar cascade files.")
        return None

    # Convert image to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=2, minSize=(20, 20))
    if len(faces) == 0:
        logger.info("No faces detected, using image center as fallback.")
        return None

    # Use the first detected face
    (x, y, w, h) = faces[0]
    pad = int(0.1 * h)  # 10% padding
    y1 = max(0, y - pad)
    y2 = min(gray.shape[0], y + h + pad)
    x1 = max(0, x - pad)
    x2 = min(gray.shape[1], x + w + pad)
    face_roi = gray[y1:y2, x1:x2]

    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor=1.01, minNeighbors=1, minSize=(5, 5))
    if len(eyes) >= 2:
        centers = []
        for (ex, ey, ew, eh) in eyes[:2]:
            centers.append([x1 + ex + ew // 2, y1 + ey + eh // 2])
        eye_center = np.mean(centers, axis=0).astype(int)
        logger.info(f"Both eyes detected, using midpoint: {eye_center}")
        return np.array([eye_center])
    elif len(eyes) == 1:
        (ex, ey, ew, eh) = eyes[0]
        eye_center_x = x1 + ex + ew // 2
        eye_center_y = y1 + ey + eh // 2
        logger.info(f"One eye detected at center: ({eye_center_x}, {eye_center_y})")
        return np.array([[eye_center_x, eye_center_y]])
    else:
        logger.info("No eyes detected, using image center as fallback.")
        return None

def segment_image_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_upload = form.save()
            image_path = os.path.join(settings.MEDIA_ROOT, image_upload.image.name)
            
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Detect eye center
            input_point = detect_eye_center(image_array)
            if input_point is None:
                # Fallback to image center if no eye is detected
                input_point = np.array([[image_array.shape[1] // 2, image_array.shape[0] // 2]])
                logger.info("Using image center as segmentation point.")
            input_label = np.array([1])  # Positive point

            # Set the image in the predictor
            predictor = load_sam2_model()
            predictor.set_image(image_array)

            # Perform segmentation
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            max_score = np.argmax(scores)
            filtered_masks = masks[max_score]
            if filtered_masks.dtype != np.uint8:
                filtered_masks = (filtered_masks * 255).astype(np.uint8)

            image_rgba = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
            image_rgba[:, :, :3] = image_array  # Copy RGB channels
            image_rgba[:, :, 3] = 0  # Set alpha channel to 0 (transparent)

            # Apply the mask to keep only the segmented area
            mask_binary = filtered_masks > 0
            masked_pixels = image_array[mask_binary]
            rgba_pixels = np.column_stack((masked_pixels, np.full(masked_pixels.shape[0], 255, dtype=np.uint8)))
            image_rgba[mask_binary] = rgba_pixels

            # Convert back to PIL Image for display
            result_image = Image.fromarray(image_rgba)

            # Save segmented image
            segmented_path = os.path.join(settings.MEDIA_ROOT, 'segmented', f'segmented_{image_upload.id}.png')
            os.makedirs(os.path.dirname(segmented_path), exist_ok=True)
            result_image.save(segmented_path)

            # Update model
            image_upload.result_image = f'segmented/segmented_{image_upload.id}.png'
            image_upload.save()
            
            return redirect('segmentation_result', image_id=image_upload.id)
    else:
        form = ImageUploadForm()
    return render(request, 'segmentation/upload.html', {'form': form})

def segmentation_result_view(request, image_id):
    try:
        image_upload = ImageUpload.objects.get(id=image_id)
        
        # Get analysis results if segmentation was successful
        analysis_result = None
        if image_upload.result_image:
            try:
                # Run CNN analysis on the segmented image
                classifier = load_cnn_classifier()
                if classifier:
                    segmented_path = os.path.join(settings.MEDIA_ROOT, image_upload.result_image.name)
                    if os.path.exists(segmented_path):
                        cnn_result = classifier.predict(segmented_path, threshold_mode='balanced')
                        
                        if 'error' not in cnn_result:
                            # Format results for template
                            analysis_result = {
                                'prediction': cnn_result['prediction'].upper(),
                                'confidence': cnn_result['confidence'] * 100,  # Convert to percentage
                                'risk_level': cnn_result['risk_level'],
                                'clinical_recommendation': cnn_result['clinical_recommendation'],
                                'threshold_used': cnn_result['threshold_used'],
                                'probabilities': {
                                    'safe': cnn_result['probabilities']['safe'] * 100,
                                    'hard': cnn_result['probabilities']['hard'] * 100,
                                },
                                'notes': f"Analysis completed using {cnn_result['threshold_mode']} threshold ({cnn_result['threshold_used']:.3f})"
                            }
                        else:
                            logger.error(f"CNN analysis error: {cnn_result['error']}")
                    else:
                        logger.error(f"Segmented image not found: {segmented_path}")
                else:
                    logger.error("CNN classifier not loaded")
            except Exception as e:
                logger.error(f"Error running CNN analysis: {e}")
        
        return render(request, 'segmentation/result.html', {
            'image_upload': image_upload,
            'analysis_result': analysis_result
        })
        
    except ImageUpload.DoesNotExist:
        return render(request, 'segmentation/upload.html', {
            'form': ImageUploadForm(),
            'error': 'Image not found'
        })

@api_view(['POST'])
def api_segment_image(request):
    """Original API endpoint - kept for backward compatibility"""
    form = ImageUploadForm(request.POST, request.FILES)
    if form.is_valid():
        image_upload = form.save()
        image_path = os.path.join(settings.MEDIA_ROOT, image_upload.image.name)
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            height, width = image_array.shape[:2]

            # Detect eye center
            input_point = detect_eye_center(image_array)
            if input_point is None:
                # Fallback to image center if no eye is detected
                input_point = np.array([[width // 2, height // 2]])
                logger.info("Using image center as segmentation point.")
            input_label = np.array([1])

            # Set the image in the predictor
            predictor = load_sam2_model()
            predictor.set_image(image_array)

            # Perform segmentation
            masks, scores, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            if masks.size > 0:
                max_score = np.argmax(scores)
                filtered_masks = masks[max_score]
                if filtered_masks.dtype != np.uint8:
                    filtered_masks = (filtered_masks * 255).astype(np.uint8)
                image_rgba = np.zeros((height, width, 4), dtype=np.uint8)
                image_rgba[:, :, :3] = image_array
                image_rgba[:, :, 3] = 0
                mask_binary = filtered_masks > 0
                masked_pixels = image_array[mask_binary]
                rgba_pixels = np.column_stack((masked_pixels, np.full(masked_pixels.shape[0], 255, dtype=np.uint8)))
                image_rgba[mask_binary] = rgba_pixels
                segmented_image = Image.fromarray(image_rgba, mode='RGBA')
                segmented_path = os.path.join(settings.MEDIA_ROOT, 'segmented', f'segmented_{image_upload.id}.png')
                os.makedirs(os.path.dirname(segmented_path), exist_ok=True)
                segmented_image.save(segmented_path)
                image_upload.result_image = f'segmented/segmented_{image_upload.id}.png'
                image_upload.save()
                logger.info(f"API segmented image saved: {image_upload.result_image.url}")
                img_buffer = io.BytesIO()
                segmented_image.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                return JsonResponse({
                    'segmented_image_url': image_upload.result_image.url,
                    'image_data': img_buffer.getvalue().hex(),
                })
            else:
                logger.error("API: No objects detected")
                image_upload.delete()
                return HttpResponse("No objects detected", status=400)
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            image_upload.delete()
            return HttpResponse(f"Error: {str(e)}", status=500)
    return HttpResponse("Invalid form data", status=400)

@api_view(['POST'])
def api_segment_and_analyze(request):
    """NEW ENDPOINT: Combined SAM2 segmentation + CNN analysis"""
    form = ImageUploadForm(request.POST, request.FILES)
    if not form.is_valid():
        return JsonResponse({'error': 'Invalid form data'}, status=400)
    
    image_upload = form.save()
    image_path = os.path.join(settings.MEDIA_ROOT, image_upload.image.name)
    segmented_path = None
    
    try:
        # Step 1: Load and validate image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        height, width = image_array.shape[:2]
        logger.info(f"Processing image: {width}x{height}")

        # Step 2: SAM2 Segmentation
        input_point = detect_eye_center(image_array)
        if input_point is None:
            input_point = np.array([[width // 2, height // 2]])
            logger.info("Using image center as segmentation point.")
        input_label = np.array([1])

        predictor = load_sam2_model()
        predictor.set_image(image_array)

        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        
        if masks.size == 0:
            raise Exception("No segmentation masks generated")

        # Process best mask
        max_score = np.argmax(scores)
        filtered_masks = masks[max_score]
        if filtered_masks.dtype != np.uint8:
            filtered_masks = (filtered_masks * 255).astype(np.uint8)

        # Create segmented image with transparency
        image_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        image_rgba[:, :, :3] = image_array
        image_rgba[:, :, 3] = 0
        
        mask_binary = filtered_masks > 0
        masked_pixels = image_array[mask_binary]
        rgba_pixels = np.column_stack((masked_pixels, np.full(masked_pixels.shape[0], 255, dtype=np.uint8)))
        image_rgba[mask_binary] = rgba_pixels
        
        segmented_image = Image.fromarray(image_rgba, mode='RGBA')
        segmented_path = os.path.join(settings.MEDIA_ROOT, 'segmented', f'segmented_{image_upload.id}.png')
        os.makedirs(os.path.dirname(segmented_path), exist_ok=True)
        segmented_image.save(segmented_path)
        
        image_upload.result_image = f'segmented/segmented_{image_upload.id}.png'
        image_upload.save()
        
        logger.info(f"Segmentation completed: {segmented_path}")

        # Step 3: CNN Analysis on segmented image
        classifier = load_cnn_classifier()
        if classifier is None:
            raise Exception("CNN classifier not available")
        
        # Get threshold mode from request
        threshold_mode = request.POST.get('threshold_mode', 'balanced')
        if threshold_mode not in ['conservative', 'balanced', 'precise']:
            threshold_mode = 'balanced'
        
        cnn_result = classifier.predict(segmented_path, threshold_mode=threshold_mode)
        
        if 'error' in cnn_result:
            raise Exception(f"CNN analysis failed: {cnn_result['error']}")
        
        logger.info(f"CNN analysis completed: {cnn_result['prediction']} with {cnn_result['confidence']:.3f} confidence")

        # Step 4: Prepare response for Flutter
        img_buffer = io.BytesIO()
        segmented_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        segmented_image_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        # Format response matching your Flutter expectations
        response_data = {
            'segmented_image_data': segmented_image_b64,
            'segmented_image_url': image_upload.result_image.url,
            'analysis_result': {
                'safety': cnn_result['prediction'] == 'safe',
                'confidence': cnn_result['confidence'],
                'recommendation': cnn_result['clinical_recommendation'],
                'risk_level': cnn_result['risk_level'],
                'probabilities': cnn_result['probabilities'],
                'threshold_used': cnn_result['threshold_used'],
                'threshold_mode': threshold_mode,
                'prediction': cnn_result['prediction'].upper(),
                'notes': f"AI analysis based on segmented facial features. Model prediction: {cnn_result['prediction']} with {cnn_result['confidence']:.1%} confidence using {threshold_mode} threshold ({cnn_result['threshold_used']:.3f})."
            },
            'model_info': {
                'sam2_version': 'SAM2.1 Hiera Tiny',
                'cnn_version': 'MobileViT-XXS',
                'analysis_timestamp': image_upload.uploaded_at.isoformat()
            }
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        logger.error(f"Combined API error: {str(e)}")
        
        # Cleanup on error
        if image_upload:
            image_upload.delete()
        if segmented_path and os.path.exists(segmented_path):
            os.remove(segmented_path)
            
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])
def api_health_check(request):
    """Health check endpoint"""
    try:
        # Check SAM2 model
        sam2_status = load_sam2_model() is not None
        
        # Check CNN model
        cnn_status = load_cnn_classifier() is not None
        
        return JsonResponse({
            'status': 'healthy' if sam2_status and cnn_status else 'degraded',
            'sam2_loaded': sam2_status,
            'cnn_loaded': cnn_status,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return JsonResponse({
            'status': 'unhealthy',
            'error': str(e)
        }, status=500)