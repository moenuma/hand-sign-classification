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

def softmax(z):
    max_z = np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z - max_z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def predict(X, W, b, t=None):
    z = np.dot(X, W) + b
    y = softmax(z)

    t_hat = np.argmax(y, axis=1)
    t_one_hot = one_hot_encode(t)
    error = y - t_one_hot

    loss = np.mean(-np.sum(t_one_hot * np.log(y + 1e-16), axis=1))

    t = t.reshape(-1)
    acc = np.mean(t == t_hat)
    
    return loss, error, acc

def one_hot_encode(labels):
    num_samples = labels.shape[0]
    t_one_hot = np.zeros((num_samples, N_class))
    t_one_hot[np.arange(num_samples), labels.flatten()] = 1
    t_one_hot = t_one_hot.reshape(-1, N_class)
    return t_one_hot

def train(X_train, t_train, X_val, t_val):
    N_train, N_feature = X_train.shape
    N_val = X_val.shape[0]

    # initialization
    W = np.zeros([N_feature, N_class])
    b = np.zeros(N_class)

    train_losses = []
    valid_accs = []

    W_best = None
    b_best = None
    acc_best = 0
    epoch_best = 0

    for epoch in range(epochs):

        loss_this_epoch = 0
        for batch in range(int(np.ceil(N_train/batch_size))):
            X_batch = X_train[batch*batch_size: (batch+1)*batch_size]
            t_batch = t_train[batch*batch_size: (batch+1)*batch_size]

            loss, error, _ = predict(X_batch, W, b, t_batch)

            loss_this_epoch += loss

            dW = (1/batch_size) * np.dot(X_batch.T, error)
            db = (1/batch_size) * np.sum(error)

            W -= alpha * dW
            b -= alpha * db

        train_loss = loss_this_epoch / (N_train/batch_size)
        train_losses.append(train_loss)

        _, _, acc = predict(X_val, W, b, t_val)
        valid_accs.append(acc)
        if acc > acc_best:
            epoch_best = epoch
            acc_best = acc
            W_best = W
            b_best = b
            
    return epoch_best, acc_best, W_best, b_best, train_losses, valid_accs

def main():
    epoch_best, acc_best, W_best, b_best, train_losses, valid_accs = train(X_train, y_train, X_val, y_val)
    _, _, acc_test = predict(X_test, W_best, b_best, y_test)
    print("The test performance (accuracy): ", acc_test)

    # Plot the learning curve of the training loss
    plt.figure(1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', marker='o', markersize=4)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Over Epoch')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./graphs/Loss_Softmax.jpg')

    # Plot the learning curve of the validation risk
    plt.figure(2)
    plt.plot(range(1, len(valid_accs) + 1), valid_accs, label='Validation Accuracy', marker='o', color='r', markersize=4)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Over Epoch')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./graphs/Accuracy_Softmax.jpg')

X_train, y_train, X_val, y_val, X_test, y_test, N_class = readMNISTdata()
epochs = 150
batch_size = 50
alpha = 0.1
main()
