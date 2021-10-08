import numpy as np
import tensorflow
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet_v2 import ResNet152V2
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout
from keras.optimizers import adam_v2
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

img_rows, img_cols = 224, 224
classes = 2
train_dir = "/Users/dipit/FaceMaskDetection/train"
test_dir = "/Users/dipit/FaceMaskDetection/test"
batch_size = 32


def preprocess_image(train_dir, test_dir, img_rows, img_cols, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical"
    )
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        shuffle=False,
        class_mode="categorical"
    )
    return train_generator, test_generator


train_data, test_data = preprocess_image(train_dir, test_dir, img_rows, img_cols, batch_size)


def build_model():
    basemodel = ResNet152V2(
        weights="imagenet",
        input_shape=(img_rows, img_cols, 3),
        include_top=False
    )
    for layer in basemodel.layers:
        layer.trainable = False
    x = basemodel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(classes, activation="softmax")(x)
    model = keras.models.Model(inputs=basemodel.input, outputs=output)
    return model


model = build_model()
print(model.summary())
model.compile(
    loss="categorical_crossentropy",
    optimizer=adam_v2.Adam(learning_rate=0.0001),
    metrics=['acc']
)


history = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data
)


print("Evaluating models")
scores = model.evaluate(test_data, verbose=1)
print("Loss: ", scores[0])
print("Accuracy: ", scores[1])


keras.models.save_model(model, "FaceMaskDetector.hdf5")
model = keras.models.load_model("FaceMaskDetector.hdf5")


print("Test Prediction")
test_pred = model.predict(test_data, verbose=1)
test_labels = np.argmax(test_pred, axis=1)

class_labels = test_data.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(classes)
print("Confusion Matrix")
print(confusion_matrix(test_data.classes, test_labels))
print("Classification Report")
print(classification_report(test_data.classes, test_labels, target_names=classes))