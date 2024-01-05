#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

def readMNISTdata():
    train_data = pd.read_csv('./dataset/sign_mnist_train.csv')
    test_data = pd.read_csv('./dataset/sign_mnist_test.csv')
    train_labels = train_data['label'].values
    test_labels = test_data['label'].values
    train_data.drop('label', axis=1, inplace=True)
    test_data.drop('label', axis=1, inplace=True)
    X_train_val = train_data.values
    X_test = test_data.values

    # Reshape the features from 1D to 3D (28x28 pixels grayscale)
    X_train_val = X_train_val.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    # Normalize the data
    X_train_val = X_train_val / 255
    X_test = X_test / 255

    # Convert the integer labels to binary (24 classes)
    lb = LabelBinarizer()
    y_train_val = lb.fit_transform(train_labels)
    y_test = lb.fit_transform(test_labels)
    
    # 20000 train samples, 7455 validation samples
    X_train = X_train_val[:20000]
    X_val = X_train_val[20000:]
    y_train = y_train_val[:20000]
    y_val = y_train_val[20000:]

    N_class = len(np.unique(np.array(train_labels)))

    return X_train, y_train, X_val, y_val, X_test, y_test, N_class

def model_CNN():
    model = Sequential()

    # Conv layer
    model.add(Conv2D(
        64,
        kernel_size=(3,3),
        input_shape=(28,28,1),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())

    # MaxPooling layer
    model.add(MaxPooling2D())

    model.add(Conv2D(
        32,
        kernel_size=(2,2),
        padding="same",
        activation="relu"
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())

    # Convert to 1D
    model.add(Flatten())

    # Dense layer
    model.add(Dense(128, activation="relu"))

    # Output layer
    model.add(Dense(N_class, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def train(X_train, y_train, X_val, y_val, model):
    history = model.fit(X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data = (X_val, y_val)
    )
    return history, model

def evaluate(X_test, y_test, model):
    (ls, acc) = model.evaluate(x=X_test, y=y_test)
    return ls, acc

def main():
    model = model_CNN()
    history, model = train(X_train, y_train, X_val, y_val, model)
    ls_test, acc_test = evaluate(X_test, y_test, model)
    print("The test performance (accuracy): ", acc_test)
    
    plt.figure(1)
    plt.plot(history.history["loss"], marker='o', markersize=4)
    plt.plot(history.history["val_loss"], marker='o', color='r', markersize=4)
    plt.title("Loss Over Epoch")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend(["training", "validation"])
    plt.grid(True)
    plt.savefig('./graphs/Loss_CNN.jpg')

    plt.figure(2)
    plt.plot(history.history["accuracy"], marker='o', markersize=4)
    plt.plot(history.history["val_accuracy"], marker='o', color='r', markersize=4)
    plt.title("Accuracy Over Epoch")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.legend(["training", "validation"])
    plt.grid(True)
    plt.savefig('./graphs/Accuracy_CNN.jpg')

X_train, y_train, X_val, y_val, X_test, y_test, N_class = readMNISTdata()
epochs = 20
batch_size = 200
main()