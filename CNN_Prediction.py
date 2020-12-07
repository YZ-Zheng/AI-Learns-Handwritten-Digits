import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class Flatten(nn.Module):
    """
    performs the flatten operation
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN(nn.Module):
    """
    duplicating the CNN architecture
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
          nn.Conv2d(1, 32, (5, 5), padding=(2, 2)),
          nn.ReLU(),
          nn.MaxPool2d((2, 2)),
          nn.Conv2d(32, 64, (3, 3), padding=(1, 1)),
          nn.ReLU(),
          nn.MaxPool2d((2, 2)),
          Flatten(),
          nn.Linear(64*7*10, 128),
          nn.Dropout(0.6),
          nn.Linear(128, 20)
        )

    def forward(self, x):
        output = self.model(x)
        out_first_digit = output[:, :10]
        out_second_digit = output[:, 10:]

        return out_first_digit, out_second_digit


model = torch.load('./CNN_Model/CNN.pth')   # load model
model.eval()


def predict():
    """
    predict the two handwritten digits

    """
    image = Image.open('numbers.jpg').convert('L')
    image = image.resize((28, 42))
    image = np.array(image)
    image = image.reshape(1, 1, 42, 28)
    image = torch.tensor(image, dtype=torch.float32)
    out1, out2 = model(image)
    predictions_first_label = torch.argmax(out1, dim=1)
    predictions_second_label = torch.argmax(out2, dim=1)

    return predictions_first_label.numpy(), predictions_second_label.numpy()
