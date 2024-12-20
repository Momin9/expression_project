import os
import shutil
import numpy as np
from PIL import Image
from deepface import DeepFace
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
import requests

# Paths for training data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, '../data/train/')
os.makedirs(MEDIA_DIR, exist_ok=True)  # Ensure MEDIA_DIR exists

# Emotion mappings
emoji_map = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}

# Helper Functions
def detect_features(image):
    """
    Use DeepFace to detect age, gender, emotion, and race from the image.
    """
    try:
        result = DeepFace.analyze(
            img_path=image,
            actions=['age', 'gender', 'emotion', 'race'],
            enforce_detection=False
        )

        # Extract key details
        features = {
            "age": result[0]['age'],
            "gender": result[0]['gender'],
            "emotion": result[0]['dominant_emotion'],
            "race": result[0]['dominant_race'],
            "emoji": emoji_map.get(result[0]['dominant_emotion'], "")
        }
        return features
    except Exception as e:
        return {"error": str(e)}

def online_train(image_path, emotion):
    """
    Save the image to the appropriate emotion directory for self-training.
    """
    try:
        emotion_dir = os.path.join(MEDIA_DIR, emotion)
        os.makedirs(emotion_dir, exist_ok=True)
        save_path = os.path.join(emotion_dir, os.path.basename(image_path))
        shutil.move(image_path, save_path)
    except Exception as e:
        print(f"Error during self-training: {str(e)}")

# View Functions
def index(request):
    """
    Render the landing page.
    """
    return render(request, 'index.html')

def image_detection(request):
    """
    Handle image uploads and detect age, gender, emotion, and race.
    """
    if request.method == 'POST':
        try:
            # Check if an image file is provided
            if 'image' not in request.FILES:
                return JsonResponse({'status': 'error', 'message': 'No image provided'})

            image = request.FILES['image']
            image_path = os.path.join(MEDIA_DIR, 'uploaded_image.jpg')

            # Save the uploaded image temporarily
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Detect features
            features = detect_features(image_path)

            if "error" in features:
                return JsonResponse({'status': 'error', 'message': features['error']})

            # Optional self-training based on emotion
            emotion = features.get('emotion')
            if emotion:
                online_train(image_path, emotion)

            return JsonResponse({'status': 'success', **features})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return render(request, 'image_detection.html')

def live_detection(request):
    """
    Render live detection page.
    """
    return render(request, 'live_detection.html')

def capture_expression(request):
    """
    Handle POST requests for live detection.
    """
    if request.method == 'POST':
        try:
            # Check if an image file is uploaded
            if 'image' in request.FILES:
                image_data = request.FILES['image']
            elif 'camera_image' in request.FILES:  # Check if a camera image is provided
                image_data = request.FILES['camera_image']
            else:
                return JsonResponse({'error': 'No image provided'})

            image_path = os.path.join(MEDIA_DIR, 'captured_image.jpg')

            # Save the captured image temporarily
            with open(image_path, 'wb+') as destination:
                for chunk in image_data.chunks():
                    destination.write(chunk)

            # Detect features
            features = detect_features(image_path)

            if "error" in features:
                return JsonResponse({'status': 'error', 'message': features['error'], "status_code": 400})

            # Optional self-training based on emotion
            emotion = features.get('emotion')
            if emotion:
                online_train(image_path, emotion)

            return JsonResponse({'status': 'success', **features})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e), "status_code": 400})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method', "status_code": 405})

def get_video_url(request):
    """
    Stream a video from an external URL.
    """
    video_url = "https://player.castr.com/vod/KOEdOwHqSL1e960u"
    response = requests.get(video_url, stream=True)
    return StreamingHttpResponse(response.iter_content(chunk_size=8192), content_type="video/mp4")
