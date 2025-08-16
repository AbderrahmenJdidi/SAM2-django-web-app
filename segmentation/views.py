import os
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from .forms import ImageUploadForm
from .models import ImageUpload
from PIL import Image
import numpy as np
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import torch
import logging
from rest_framework.decorators import api_view
import io
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

predictor = None

def load_sam2_model():
    global predictor
    if predictor is None:
        checkpoint = "C:\\Users\\jdidi\\OneDrive\\Bureau\\stage_2eme\\application\\sam2_django_app\\segment-anything-2\\checkpoints\\sam2.1_hiera_tiny.pt"
        model_cfg = "C:\\Users\\jdidi\\OneDrive\\Bureau\\stage_2eme\\application\\sam2_django_app\\segment-anything-2\\sam2\\configs\\sam2.1\\sam2.1_hiera_t.yaml"
        device = torch.device("cpu")
        sam2_model = build_sam2(model_cfg, checkpoint, device)
        predictor = SAM2ImagePredictor(sam2_model)
    return predictor

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
        return render(request, 'segmentation/result.html', {'image_upload': image_upload})
    except ImageUpload.DoesNotExist:
        return render(request, 'segmentation/upload.html', {
            'form': ImageUploadForm(),
            'error': 'Image not found'
        })

@api_view(['POST'])
def api_segment_image(request):
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