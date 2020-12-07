import tkinter as tk
import torch
import torch.nn as nn
import PIL
from PIL import ImageGrab, Image
import numpy as np

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


model = torch.load('./CNN_Model/CNN.pth') # load model
model.eval()


class Draw:
    """
    creating a drawing tool using TK interface
    """
    def __init__(self, master):
        self.master = master
        self.old_x = None
        self.old_y = None
        self.penwidth = 26
        self.paint()
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.reset)

    def draw(self, p):
        """
        create white pixels when left mouse button is clicked

        """
        if self.old_x and self.old_y:
            self.canvas.create_line(self.old_x, self.old_y, p.x, p.y, width=self.penwidth, fill='white',
                                    capstyle='round', smooth=True)
        self.old_x = p.x
        self.old_y = p.y

    def reset(self, p):
        self.old_x = None
        self.old_y = None

    def predict(self):
        """
        predict the two digits drawn

        """
        image = Image.open('numbers.jpg').convert('L')
        image = image.resize((28, 42))
        image = np.array(image, dtype=float)
        image = image.reshape((1, 1, 42, 28))
        image = torch.tensor(image, dtype=torch.float32)
        out1, out2 = model(image)
        predictions_first_label = torch.argmax(out1, dim=1)
        predictions_second_label = torch.argmax(out2, dim=1)

        return predictions_first_label.numpy(), predictions_second_label.numpy()

    def getNum(self):
        """
        screenshot the drawing and save it as numbers.jpg, may need to adjust the x and y values so that it actually
        captures the right portion of the screen

        """
        x = self.master.winfo_rootx() + self.canvas.winfo_x() + 160  # may need to adjust this value to capture the right portion of the screen
        y = self.master.winfo_rooty() + self.canvas.winfo_y() + 17   # may need to adjust this value to capture the correct portiorn of the screen
        x1 = x + self.canvas.winfo_width() + 60   # may need adjust this too
        y1 = y + self.canvas.winfo_height() + 105  # may need adjust this too
        img = PIL.ImageGrab.grab((x, y, x1, y1))
        img.save("numbers.jpg")
        self.result1, self.result2 = self.predict()
        self.uppernumber['text'] = "Upper number: " + str(int(self.result1))
        self.lowernumber['text'] = "Lower number: " + str(int(self.result2))

    def paint(self):
        self.canvas = tk.Canvas(master=self.master, width=280, height=420, bg='black', cursor='circle')
        self.canvas.grid(row=0, column=0)
        self.frame = tk.Frame(master=self.master, pady=10)
        self.frame.grid(row=1, column=0)
        self.uppernumber = tk.Label(master=self.frame, text="Upper number: None", font=100)
        self.uppernumber.grid(pady=5)
        self.lowernumber = tk.Label(master=self.frame, text="Lower number: None", font=100)
        self.lowernumber.grid(pady=5)
        tk.Button(master=self.frame, text="Display Numebers", font=100, command=self.getNum).grid(ipady=5, padx=10, pady=10)
        tk.Button(master=self.frame, text="Clear Canvas", command=self.clear, font=100).grid(ipadx=15)

    def clear(self):
        self.canvas.delete('all')


if __name__ == '__main__':
    window = tk.Tk()
    window.columnconfigure(0, weight=1)
    window.rowconfigure([0, 1], weight=1)
    Draw(window)
    window.title("Drawing tool")
    window.mainloop()
