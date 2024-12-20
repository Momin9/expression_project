import os

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set directories for training and validation data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "../data")  # Go one level up to locate "data"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VALID_DIR = os.path.join(DATASET_DIR, "test")

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
NUM_CLASSES = 7  # Seven expressions

# Data augmentation for training
data_gen_train = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

data_gen_val = ImageDataGenerator(rescale=1.0 / 255)

# Load the training and validation data
generator_train = data_gen_train.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)

generator_val = data_gen_val.flow_from_directory(
    VALID_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
)


# Define the ResNet50 model
def create_model():
    base_model = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights="imagenet")
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)  # Add dropout for regularization
    x = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=x)

    # Fine-tune the top layers
    for layer in base_model.layers[:-10]:  # Freeze all but the last 10 layers
        layer.trainable = False

    return model


# Compile the model
model = create_model()
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    generator_train,
    epochs=EPOCHS,
    validation_data=generator_val,
    steps_per_epoch=len(generator_train),
    validation_steps=len(generator_val),
)

# Save the trained model
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "expression_model_resnet1.h5")
model.save(MODEL_SAVE_PATH)
print(f"Model saved at {MODEL_SAVE_PATH}")

# Evaluate the model
train_loss, train_acc = model.evaluate(generator_train)
val_loss, val_acc = model.evaluate(generator_val)
print(f"Training Accuracy: {train_acc:.2f}, Validation Accuracy: {val_acc:.2f}")
