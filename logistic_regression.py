#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def readMNISTdata():
    train_data = pd.read_csv('./dataset/sign_mnist_train.csv')
    test_data = pd.read_csv('./dataset/sign_mnist_test.csv')
    y_train_val = train_data['label'].values
    y_test = test_data['label'].values
    train_data.drop('label', axis=1, inplace=True)
    test_data.drop('label', axis=1, inplace=True)
    X_train_val = train_data.values
    X_test = test_data.values

    # Normalize the data
    X_train_val = X_train_val / 255
    X_test = X_test / 255
    
    # 20000 train samples, 7455 validation samples
    X_train = X_train_val[:20000]
    X_val = X_train_val[20000:]
    y_train = y_train_val[:20000]
    y_val = y_train_val[20000:]

    class_min = min(y_train_val)
    class_max = max(y_train_val)
    N_class = class_max - class_min + 1

    return X_train, y_train, X_val, y_val, X_test, y_test, N_class

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X_train, t_train, X_val, t_val, label):
    # Set the binary labels for the current class vs. all other classes
    t_train_binary = np.where(t_train == label, 1, 0)
    t_val_binary = np.where(t_val == label, 1, 0)
    
    # Initialize weights and bias
    N_train, N_feature = X_train.shape
    w = np.zeros(N_feature)
    b = 0

    train_losses = []
    valid_accs = []

    w_best = None
    b_best = None
    acc_best = 0
    epoch_best = 0
    
    for epoch in range(epochs):

        loss_this_epoch = 0
        for batch in range(int(np.ceil(N_train/batch_size))):
            X_batch = X_train[batch*batch_size: (batch+1)*batch_size]
            t_batch = t_train_binary[batch*batch_size: (batch+1)*batch_size]

            loss, error, _ = predict(X_batch, w, b, t_batch)
            loss_this_epoch += loss
            
            dw = (1/batch_size) * np.dot(X_batch.T, error)
            db = (1/batch_size) * np.sum(error)

            w -= alpha * dw
            b -= alpha * db
        
        train_loss = loss_this_epoch / (N_train/batch_size)
        train_losses.append(train_loss)

        _, _, acc = predict(X_val, w, b, t_val_binary)
        valid_accs.append(acc)
        if acc > acc_best:
            epoch_best = epoch
            acc_best = acc
            w_best = w
            b_best = b
    
    return epoch_best, acc_best, w_best, b_best, train_losses, valid_accs

def computeLoss(t, y):
    return -np.mean(t * np.log(y+1e-16) + (1-t) * np.log(1-y+1e-16))

def predict(X, w, b, t):
    z = np.dot(X, w) + b
    y = sigmoid(z)
    error = y - t
    loss = -np.mean(t * np.log(y+1e-16) + (1-t) * np.log(1-y+1e-16))
    t_hat = (y >= 0.5).astype(int)
    acc = np.mean(t == t_hat)
    return loss, error, acc


def predict_one_vs_all(classifiers, X, t):
    y = np.zeros((len(X), len(classifiers)))

    for label, (w, b) in classifiers.items():
        z = np.dot(X, w) + b
        y[:, label] = sigmoid(z)

    t_hat = np.argmax(y, axis=1)
    acc = np.mean(t_hat == t)
    return acc


def main():
    classifiers = {}
    loss_data = {}
    acc_data = {}

    for label in range(N_class):
        epoch_best, acc_best, w_best, b_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val, label)
        print(acc_best)
        classifiers[label] = (w_best, b_best)
        loss_data[label] = train_losses
        acc_data[label] = valid_accs

    acc_test = predict_one_vs_all(classifiers, X_test, t_test)
    print("The test performance (accuracy): ", acc_test)

    plt.figure(1)
    for label, loss_data in loss_data.items():
        plt.plot(loss_data, label=label)
    plt.title("Training Loss Over Epoch")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Training Loss")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=8)
    plt.tight_layout()
    plt.savefig('./graphs/Loss_Logistic_Regression.jpg')

    plt.figure(2)
    for label, acc_data in acc_data.items():
        plt.plot(acc_data, label=label)
    plt.title("Validation Accuracy Over Epoch")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=8)
    plt.tight_layout()
    plt.savefig('./graphs/Accuracy_Logistic_Regression.jpg')

# Global variables
X_train, t_train, X_val, t_val, X_test, t_test, N_class = readMNISTdata()
epochs = 30
batch_size = 100
alpha = 0.1

main()