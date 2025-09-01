# AnesthesiaSafe - Django Web App

Django backend service providing AI-powered image segmentation and analysis for pediatric anesthesia safety assessment using SAM2 and CNN models.

## Overview

This Django application serves as the backend API for the AnesthesiaSafe mobile application. It processes medical images using the SAM2 (Segment Anything Model 2) for image segmentation and integrates with CNN models for safety analysis in pediatric anesthesia procedures.

AnesthesiaSafe repository URL : https://github.com/AbderrahmenJdidi/anesthesiasafe


## Installation & Setup

### Prerequisites
- Python 3.8+
- SAM2 model files and configuration
- OpenCV
- PyTorch (CPU or GPU version)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AbderrahmenJdidi/SAM2-django-web-app
   cd sam2_django_backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```


4. **Database Setup**
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

5. **Create Media Directories**
   ```bash
   mkdir -p media/uploads media/segmented
   ```

6. **Run the server**
   ```bash
   python manage.py runserver 0.0.0.0:8000
   ```

## API Endpoints

### Image Segmentation API
```http
POST /api/segment-and-analyze/
Content-Type: multipart/form-data

Parameters:
- image: Image file (JPEG, PNG)

Response:
{
  "segmented_image_url": "/media/segmented/segmented_123.png",
  "image_data": "hexadecimal_image_data"
}
```

### Health Check API
```http
GET /api/health/

Response:
200 OK - Server is running
```

### Web Interface
- `/` - Image upload and processing interface
- `/result/<int:image_id>/` - View segmentation results


### OpenCV Configuration
- Uses pre-trained Haar cascades for face/eye detection
- Configurable detection parameters for accuracy
- Fallback mechanisms for edge cases

## Database Models

### ImageUpload Model
```python
class ImageUpload(models.Model):
    image = models.ImageField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    result_image = models.ImageField(upload_to='segmented/', null=True, blank=True)
```

## Configuration

### Django Settings
```python
# Network configuration
ALLOWED_HOSTS = ['127.0.0.1', '192.168.1.11', '0.0.0.0', 'localhost']

# Media files
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Apps
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'segmentation.apps.SegmentationConfig',
    'rest_framework',
]
```

## Running the Server

### Development Server
```bash
# Standard development server
python manage.py runserver

# Listen on all interfaces (required for mobile app)
python manage.py runserver 0.0.0.0:8000

# Custom port
python manage.py runserver 0.0.0.0:8080
```

## Testing

### Web Interface Testing
1. Navigate to `http://127.0.0.1:8000/`
2. Upload a test image
3. View segmentation results
4. Check generated files in media directories



### Mobile App Integration
- Ensure server is accessible from mobile device network
- Test image upload from Flutter app
- Verify response format compatibility

#
