import os

from keras.applications import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model


# Load the model globally
def load_expression_model():
    """
    Load the MobileNet-based model for facial expression detection.
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    weights_path = os.path.join(BASE_DIR, 'mobilenet_1_0_224_tf_no_top.h5')

    base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    # Add custom layers for emotion detection
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(7, activation='softmax')(x)  # Assuming 5 emotion classes

    model = Model(inputs=base_model.input, outputs=x)

    # Load pre-trained weights (ensure compatibility)
    model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    return model


# Initialize the model
model = load_expression_model()
