from django.shortcuts import render
from django.http import JsonResponse
import cv2
import numpy as np
from expression_model.Final_Expression_detection import load_expression_model

# Load the model globally
model = load_expression_model()

def preprocess_image(image):
    """
    Preprocess the image to match model training preprocessing.
    """
    image = cv2.resize(image, (224, 224))  # Resize to model input size
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize to [0, 1]
    return image

def detect_expression(image):
    """
    Detect facial expression from the given image.
    """
    image = preprocess_image(image)  # Preprocess the image

    # Predict the expression
    prediction = model.predict(image)
    print("Prediction Probabilities:", prediction)  # Debug: Print raw predictions

    expression_map = {
        0: 'happy',
        1: 'sad',
        2: 'angry',
        3: 'surprised',
        4: 'neutral'
    }
    emoji_map = {
        'happy': '😊',
        'sad': '😢',
        'angry': '😡',
        'surprised': '😲',
        'neutral': '😐'
    }

    detected_expression = expression_map[np.argmax(prediction)]
    detected_emoji = emoji_map[detected_expression]
    return detected_expression, detected_emoji

def index(request):
    """
    Renders the homepage.
    """
    return render(request, 'index.html')

def capture_expression(request):
    """
    Handle the POST request to capture an image and detect the facial expression.
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

            # Convert image data to OpenCV format
            np_arr = np.frombuffer(image_data.read(), np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Detect expression and emoji
            expression, emoji = detect_expression(image)
            return JsonResponse({'expression': expression, 'emoji': emoji})
        except Exception as e:
            return JsonResponse({'error': str(e)})

    return JsonResponse({'error': 'Invalid request method'})
