from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('capture-expression/', views.capture_expression, name='capture_expression'),
]
