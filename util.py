import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets
from torch.utils.data import DataLoader
import torch
import time

def plot_linear_boundary(X_train, Y_train, X_test, Y_test, w, b):
    x_min = min(X_train[:, 0])
    x_max = min(X_train[:, 1])
    y_min = ( (-(w[0] * x_min) - b[0]) 
            / w[1] )

    x_max = 5
    y_max = ( (-(w[0] * x_max) - b[0]) 
            / w[1] )
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax[0].scatter(X_train[Y_train == 0, 0], X_train[Y_train == 0, 1], label="class 0")
    ax[0].scatter(X_train[Y_train == 1, 0], X_train[Y_train == 1, 1], label="class 1")
    ax[0].plot([x_min, x_max], [y_min, y_max], color="black")
    ax[0].set_title("Training set")

    ax[1].scatter(X_test[Y_test == 0, 0], X_test[Y_test == 0, 1], label="class 0")
    ax[1].scatter(X_test[Y_test == 1, 0], X_test[Y_test == 1, 1], label="class 1")
    ax[1].plot([x_min, x_max], [y_min, y_max], color="black")
    ax[1].set_title("Test set")
    ax[1].legend()

    plt.show()

def compute_batch_accuracy(model, data_loader):
    correct, num_examples = 0, 0
    with torch.no_grad():
        for features, target in data_loader:
            logits = model(features)
            _, predicted = torch.max(logits, dim=1)
            num_examples += target.shape[0]
            correct += sum(predicted == target)
    acc = correct.float() / num_examples * 100
    return acc

def get_MNIST_loaders(batch_size, transform):
    train = torchvision.datasets.MNIST(
        root='data',
        transform=transform, 
        download=True
    )

    train, valid = torch.utils.data.random_split(train, [50000, 10000])

    test = torchvision.datasets.MNIST(
        root='data', 
        transform=transform, 
        train=False
    )

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        valid,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False
    )
    return train_loader, valid_loader, test_loader


def train_model(model, optimizer, scheduler, loss_fn, 
    train_loader, valid_loader, test_loader, num_epochs, batch_size):

    train_acc_ls, valid_acc_ls, mini_batch_loss_ls = [], [], []

    start_time = time.time()
    for epoch in range(num_epochs):
        for batch_idx, (features, target) in enumerate(train_loader):

            # FORWARD
            logits = model(features)

            # BACKWARD
            loss = loss_fn(logits, target)
            mini_batch_loss_ls.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            
            # UPDATE WEIGHTS
            optimizer.step()
            
            # BATCH LOGGING
            if batch_idx % 50 == 0:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                    %(epoch + 1, num_epochs, batch_idx, batch_size, loss))

        # EPOCH LOGGING
        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

        train_acc = compute_batch_accuracy(model, train_loader)
        valid_acc = compute_batch_accuracy(model, valid_loader)
        print('Epoch: %03d/%03d Train: %.2f%% | Validation: %.2f%%' % (
            epoch + 1, num_epochs, train_acc, valid_acc
        ))
        train_acc_ls.append(train_acc)
        valid_acc_ls.append(valid_acc)

        # ADJUST LEARNING RATE
        scheduler.step(valid_acc_ls[-1])

    print("-------------------------------")

    # FINAL LOGGING
    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_batch_accuracy(model, test_loader)
    print(f'Test accuracy {test_acc :.2f}%')

    return mini_batch_loss_ls, train_acc_ls, valid_acc_ls

def plot_accuracy(train_acc, valid_acc):
    fig, ax = plt.subplots()
    ax.plot(train_acc, label="train")
    ax.plot(valid_acc, label="valid")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    plt.show()

def plot_training_loss(loss, window_size=100):
    fig, ax = plt.subplots()
    mean_loss = np.convolve(loss, np.ones(window_size)/window_size, mode='valid')
    ax.plot(loss, label="Minibatch loss")
    ax.plot(mean_loss, label="Moving Average")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.show()