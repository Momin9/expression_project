from keras.applications import MobileNet
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(BASE_DIR, 'mobilenet_1_0_224_tf_no_top.h5')

def load_expression_model():
    """
    Load the MobileNet-based model for facial expression detection.
    """
    base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights=None)
    # Add custom layers for emotion detection
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(5, activation='softmax')(x)  # Assuming 5 emotion classes

    model = Model(inputs=base_model.input, outputs=x)

    # Load pre-trained weights
    return model
