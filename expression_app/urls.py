from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('live-detection/', views.live_detection, name='live_detection'),
    path('image-detection/', views.image_detection, name='image_detection'),
    path('capture-expression/', views.capture_expression, name='capture_expression'),
    path('get-video-url/', views.get_video_url, name='get_video_url'),
]
