# import os
# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from django.http import JsonResponse
# from django.shortcuts import render
#
# # Define constants
# IMG_SIZE = 224
# NUM_CLASSES = 7  # Seven expressions
# MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "expression_model.h5")
# DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
# TRAIN_DIR = os.path.join(DATASET_DIR, "train")
#
# # Load or create model
# def load_or_create_model():
#     if os.path.exists(MODEL_PATH):
#         model = load_model(MODEL_PATH)
#     else:
#         from tensorflow.keras.applications import MobileNet
#         from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
#         from tensorflow.keras.models import Model
#
#         base_model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
#         x = GlobalAveragePooling2D()(base_model.output)
#         x = Dense(NUM_CLASSES, activation="softmax")(x)
#         model = Model(inputs=base_model.input, outputs=x)
#
#         for layer in base_model.layers:
#             layer.trainable = False
#
#         model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
#     return model
#
# model = load_or_create_model()
#
# # Function to preprocess the image
# def preprocess_image(image):
#     image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = np.expand_dims(image, axis=0)
#     image = image / 255.0
#     return image
#
# # Function to detect expression
# def detect_expression(image):
#     image = preprocess_image(image)
#     prediction = model.predict(image)
#
#     expression_map = {
#         0: "angry",
#         1: "disgust",
#         2: "fear",
#         3: "happy",
#         4: "neutral",
#         5: "sad",
#         6: "surprise",
#     }
#
#     emoji_map = {
#         "angry": "😡",
#         "disgust": "🤢",
#         "fear": "😨",
#         "happy": "😊",
#         "neutral": "😐",
#         "sad": "😢",
#         "surprise": "😲",
#     }
#
#     detected_expression = expression_map[np.argmax(prediction)]
#     detected_emoji = emoji_map[detected_expression]
#     return detected_expression, detected_emoji
#
# # Function to retrain the model with new data
# def online_train(image, label):
#     image = preprocess_image(image)
#     label = to_categorical(label, NUM_CLASSES)
#     model.fit(image, np.expand_dims(label, axis=0), epochs=1, verbose=1)
#     model.save(MODEL_PATH)
#
# # API endpoint for expression detection and self-training
# def capture_expression(request):
#     if request.method == "POST":
#         try:
#             # Check if an image file is uploaded
#             if "image" in request.FILES:
#                 image_data = request.FILES["image"]
#             elif "camera_image" in request.FILES:
#                 image_data = request.FILES["camera_image"]
#             else:
#                 return JsonResponse({"error": "No image provided"})
#
#             # Convert image data to OpenCV format
#             np_arr = np.frombuffer(image_data.read(), np.uint8)
#             image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
#             # Detect expression and emoji
#             expression, emoji = detect_expression(image)
#
#             # Self-train the model with the detected expression
#             expression_map = {
#                 "angry": 0,
#                 "disgust": 1,
#                 "fear": 2,
#                 "happy": 3,
#                 "neutral": 4,
#                 "sad": 5,
#                 "surprise": 6,
#             }
#
#             online_train(image, expression_map[expression])
#
#             return JsonResponse({"expression": expression, "emoji": emoji})
#         except Exception as e:
#             return JsonResponse({"error": str(e)})
#
#     return JsonResponse({"error": "Invalid request method"})
#
# # Views for rendering templates
# def index(request):
#     return render(request, "index.html")
#
# def live_detection(request):
#     return render(request, "live_detection.html")
#
# def image_detection(request):
#     return render(request, "image_detection.html")
import os
import shutil

import numpy as np
import requests
from PIL import Image
from deepface import DeepFace
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render

# Paths for training data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, '../data/train/')

# Emotion classes
emotion_map = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}
emoji_map = {
    "angry": "😡",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😊",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲",
}


def preprocess_image(image):
    """
    Preprocess the image for DeepFace emotion detection.
    """
    image = np.array(Image.open(image).convert('RGB'))
    return image


def detect_expression(image):
    """
    Use DeepFace to detect emotion from the image.
    """
    try:
        result = DeepFace.analyze(img_path=image, actions=['emotion'], enforce_detection=False)

        # Check if result is a single dictionary or a list of dictionaries
        if isinstance(result, list):  # Multiple faces detected
            # Use the first detected face's emotion
            emotion = result[0]['dominant_emotion']
        else:  # Single face detected
            emotion = result['dominant_emotion']

        return emotion
    except Exception as e:
        return f"Error: {str(e)}"


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


def index(request):
    """
    Render the landing page.
    """
    return render(request, 'index.html')


def image_detection(request):
    """
    Handle image uploads and detect facial expressions.
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

            # Detect emotion
            emotion = detect_expression(image_path)

            if emotion not in emotion_map.values():
                return JsonResponse({'status': 'error', 'message': 'Emotion detection failed'})

            # Self-training: Save the image to the appropriate emotion directory
            online_train(image_path, emotion)

            return JsonResponse({'status': 'success', 'emotion': emotion})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)})

    return render(request, 'image_detection.html')


def live_detection(request):
    """
    Render live detection page.
    """
    return render(request, 'live_detection.html')


def get_video_url(request):
    video_url = "https://player.castr.com/vod/KOEdOwHqSL1e960u"
    response = requests.get(video_url, stream=True)
    return StreamingHttpResponse(response.iter_content(chunk_size=8192), content_type="video/mp4")

def capture_expression(request):
    """
    Handle POST requests for live detection.
    """
    if request.method == 'POST':
        try:
            # Ensure the MEDIA_DIR exists
            # os.makedirs(MEDIA_DIR, exist_ok=True)
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

            # Detect emotion
            emotion = detect_expression(image_path)

            if emotion not in emotion_map.values():
                return JsonResponse({'status': 'error', 'message': 'Emotion detection failed', "status_code": 400})

            # Self-training: Save the image to the appropriate emotion directory
            online_train(image_path, emotion)
            if emotion in emoji_map:
                emoji = emoji_map.get(emotion)

            return JsonResponse({'expression': emotion, 'emoji': emoji})

        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e), "status_code": 400})

    return JsonResponse({'status': 'error', 'message': 'Invalid request method', "status_code": 405})
