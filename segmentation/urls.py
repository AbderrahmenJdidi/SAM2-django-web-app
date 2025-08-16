from django.urls import path
from . import views

urlpatterns = [
    path('', views.segment_image_view, name='segment_image'),
    path('result/<int:image_id>/', views.segmentation_result_view, name='segmentation_result'),
    path('api/segment/', views.api_segment_image, name='api_segment_image'),
]