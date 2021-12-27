import matplotlib.pyplot as plt
import numpy as np
import torch

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

def compute_batch_accuracy(model, data_loader, num_features):
    correct, num_examples = 0, 0
    for features, target in data_loader:
        features = features.view(-1, num_features)
        _, probas = model(features)
        _, predicted = torch.max(probas, dim=1)
        num_examples += target.shape[0]
        correct += sum(predicted == target)
    return correct.float() / num_examples * 100
