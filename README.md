# Facial Expression Detection

This project is a web application for detecting facial expressions (e.g., happy, sad, angry, surprised, neutral) from an uploaded image or a camera capture. It uses a pre-trained deep learning model based on **MobileNet** for inference and is built using **Django** as the backend framework.

## Features
- Upload an image to detect facial expressions.
- Use your device's camera to capture an image and detect the expression.
- Displays the detected expression with an appropriate emoji.
- Option to clear the results and reset the form.

## Technologies Used
- **Frontend**:
  - HTML5, CSS3
  - [MDB UI Kit](https://mdbootstrap.com/) for modern styling
- **Backend**:
  - Django (Python)
  - OpenCV for image preprocessing
  - TensorFlow/Keras for deep learning inference
- **Model**:
  - Pre-trained **MobileNet** with custom layers for facial expression recognition.

## Installation

### Prerequisites
- Python 3.8+
- pip (Python package installer)
- [Virtualenv](https://virtualenv.pypa.io/en/latest/) (recommended)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/facial-expression-detection.git
   cd facial-expression-detection

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

5. Apply database migrations:
   ```bash
   python manage.py migrate
6. Open the application in your browser at:
   ```arduino
   http://127.0.0.1:8000/

## Project Structure

```plaintext
facial-expression-detection/
├── expression_project/          # Main Django project folder
│   ├── settings.py              # Django settings
│   ├── urls.py                  # URL configurations
│   ├── wsgi.py                  # WSGI configuration
│   ├── ...
├── expression_app/              # Django app folder
│   ├── templates/               # HTML templates
│   │   └── index.html           # Frontend page
│   ├── static/                  # Static files (CSS, JS, favicon)
│   ├── views.py                 # Application logic
│   ├── urls.py                  # App-specific URLs
│   ├── ...
├── expression_model/            # Model-related code
│   ├── Final_Expression_detection.py  # Model definition and loading
│   ├── mobilenet_1_0_224_tf_no_top.h5 # Pre-trained weights
├── manage.py                    # Django's management script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation


## Usage

1. **Upload Image**  
   Choose an image file from your device.  
   Click the "**Detect Expression**" button to detect the expression in the image.

2. **Capture from Camera**  
   Use your device's camera to capture an image.  
   Click "**Capture**" to take a photo.  
   Click "**Detect Expression**" to analyze the captured image.

3. **Clear Result**  
   Use the "**Clear Result**" button to reset the result and start fresh.

## Example Expressions

| Expression | Emoji   |
|------------|---------|
| Happy      | 😊       |
| Sad        | 😢       |
| Angry      | 😡       |
| Surprised  | 😲       |
| Neutral    | 😐       |

## Known Issues

- The model may produce inaccurate results if the input image is unclear or contains multiple faces.
- GPU support requires proper installation of CUDA and cuDNN.

## Future Improvements

- Train the model on a larger dataset (e.g., AffectNet, FER2013) for better accuracy.
- Add multi-face detection and expression recognition.
- Improve the UI for a better user experience.

## Troubleshooting

### GPU Not Detected
Ensure that CUDA and cuDNN are properly installed. Refer to TensorFlow's GPU support guide.

### CSRF Errors
If hosting on an external URL (e.g., ngrok), add the URL to `CSRF_TRUSTED_ORIGINS` in `settings.py`:
```python
CSRF_TRUSTED_ORIGINS = ['https://your-ngrok-url.ngrok-free.app']
## Missing Favicon Warning

Add a `favicon.ico` file to the `static` folder and include it in your template:
```html
<link rel="icon" href="{% static 'favicon.ico' %}">
## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, feel free to contact:

**Name**: Momin Ali  
**Email**: mominalikhoker589@gmail.com  
**GitHub**: [Momin9](https://github.com/Momin9)
