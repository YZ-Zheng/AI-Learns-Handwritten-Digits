import numpy as np
import torch
import torch.nn as nn
from train_data import batchify_data, run_epoch, train_model, Flatten
import load_data as U
import matplotlib.pyplot as plt
path_to_data_dir = 'Datasets/'


learning_rate = 0.001
batch_size = 128
img_rows, img_cols = 42, 28 # input image dimensions

# check if gpu is available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("GPU is available for training")
else:
    device = torch.device("cpu")
    print("GPU is unavailable for training")


class CNN(nn.Module):
    """
    constructing the CNN
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(1, 16, (3, 3), padding=(1, 1)),
          nn.ReLU(),
          nn.MaxPool2d((2, 2)),
          nn.Conv2d(16, 32, (3, 3), padding=(1, 1)),
          nn.ReLU(),
          nn.MaxPool2d((2, 2)),
          nn.Conv2d(32, 64, (5, 5), padding=(2, 2)),
          nn.ReLU(),
          nn.MaxPool2d((2, 2)),
          Flatten(),
          nn.Linear(64*5*3, 128),
          nn.Dropout(0.6),
          nn.Linear(128, 20)
        )

    def forward(self, x):
        output = self.model(x)
        out_first_digit = output[:, :10]
        out_second_digit = output[:, 10:]

        return out_first_digit, out_second_digit


def main():
    """
    train and plot

    """
    X_train, y_train, X_test, y_test = U.get_data(path_to_data_dir)

    # Split into train and dev
    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = [y_train[0][dev_split_index:], y_train[1][dev_split_index:]]
    X_train = X_train[:dev_split_index]
    y_train = [y_train[0][:dev_split_index], y_train[1][:dev_split_index]]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [[y_train[0][i] for i in permutation], [y_train[1][i] for i in permutation]]

    # Split dataset into batches
    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Load model in device
    model = CNN().to(device)

    # Train
    acc_train_upper, acc_train_lower, acc_val_upper, acc_val_lower = train_model(train_batches, dev_batches, model)

    # Evaluate the model on test data
    loss, acc = run_epoch(test_batches, model.eval(), None)
    print('Test loss1: {:.6f}  accuracy1: {:.6f}  loss2: {:.6f}   accuracy2: {:.6f}'.format(loss[0], acc[0], loss[1], acc[1]))

    # Save model
    model.to('cpu')
    torch.save(model, './CNN_Model/CNN.pth')

    # Plot results
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(range(1, len(acc_train_upper) + 1), acc_train_upper, color='#DA9111',
               label="Training accuracy upper digit", linewidth=1.6)
    ax[0].plot(range(1, len(acc_val_upper) + 1), acc_val_upper, color='#1A70A8',
               label="validation accuracy upper digit", linewidth=1.6)
    legend = ax[0].legend(loc='lower right', shadow=True, prop={'size': 18})
    ax[0].tick_params(axis='both', labelsize=16)
    ax[0].set_xlabel("Epoch", fontsize=23)
    ax[0].set_ylabel("Accuracy", fontsize=23)

    ax[1].plot(range(1, len(acc_train_lower) + 1), acc_train_lower, color='#DA9111',
               label="Training accuracy lower digit", linewidth=1.6)
    ax[1].plot(range(1, len(acc_val_lower) + 1), acc_val_lower, color='#1A70A8',
               label="Validation accuracy lower digit", linewidth=1.6)
    legend = ax[1].legend(loc='lower right', shadow=True, prop={'size': 18})
    ax[1].tick_params(axis='both', labelsize=16)
    ax[1].set_xlabel("Epoch", fontsize=23)
    ax[1].set_ylabel("Accuracy", fontsize=23)

    plt.show()


if __name__ == '__main__':
    main()
