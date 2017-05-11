import MNIST_softmax_regression
from MNIST_softmax_regression import Weights
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

array_0 = []
i = 0
while i < 784:
    array_0.append(Weights[i][0])
    i = i + 1

my_number_array = []
height = 0
while height < 28:
    line_array = []
    width = 0
    while width < 28:
        line_array.append(array_0[(height * 28) + width])
        width = width + 1
    my_number_array.append(line_array)
    height = height + 1

data = np.asarray(my_number_array)
#Rescale to 0-255 and convert to uint8
rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
im = Image.fromarray(rescaled)
im.save('images/test.png')
