import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

IMG_SIZE = (128,128)
BATCH_SIZE = 32
EPOCHS = 15

def build_cnn(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation="relu"),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model():
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=25, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_gen = train_datagen.flow_from_directory("plant_data/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
    val_gen = val_datagen.flow_from_directory("plant_data/val", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
    num_classes = train_gen.num_classes
    model = build_cnn(num_classes)
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)
    model.save("plant_model.h5")
    logging.info("Model saved as plant_model.h5")
    plot_history(history)
    evaluate_model(model, val_gen)

def plot_history(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title("Loss")
    plt.savefig("training_history.png")

def evaluate_model(model, val_gen):
    val_gen.reset()
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    report = classification_report(y_true, y_pred, target_names=list(val_gen.class_indices.keys()))
    with open("classification_report.txt","w") as f:
        f.write(report)
    logging.info("Evaluation complete. Report saved.")

app = Flask(__name__)
model = None
class_labels = None

def load_model():
    global model, class_labels
    if os.path.exists("plant_model.h5"):
        model = tf.keras.models.load_model("plant_model.h5")
        datagen = ImageDataGenerator(rescale=1./255)
        gen = datagen.flow_from_directory("plant_data/train", target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical")
        class_labels = {v:k for k,v in gen.class_indices.items()}
    else:
        logging.error("Model not found. Train first.")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error":"Model not loaded"})
    file = request.files["file"]
    if file:
        img = load_img(file, target_size=IMG_SIZE)
        x = img_to_array(img)/255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x)
        pred_class = np.argmax(preds, axis=1)[0]
        return jsonify({"prediction":class_labels[pred_class]})
    return jsonify({"error":"No file uploaded"})

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_model()
    else:
        load_model()
        app.run(debug=True)
