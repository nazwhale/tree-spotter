import MNIST_softmax_regression
from MNIST_softmax_regression import Weights
from MNIST_softmax_regression import bias
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

I = np.asarray(PIL.Image.open('images/its_a_three.png'))
I = I.flatten()

a_four = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,51,132,214,253,254,253,203,162,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,102,142,203,203,253,252,253,252,151,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,254,253,244,203,142,102,82,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,172,252,203,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,223,234,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,122,253,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,254,91,51,51,51,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,223,253,252,253,252,253,172,82,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,21,214,253,203,162,102,102,203,223,254,253,51,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61,253,171,0,0,0,0,0,20,112,192,253,212,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,102,203,234,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,213,232,82,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,62,203,234,112,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,213,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,153,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,233,212,0,0,0,0,0,0,0,0,0,0,0,0,113,92,0,0,0,0,0,0,0,0,0,0,31,173,244,40,0,0,0,0,0,0,0,0,0,0,0,82,253,151,0,0,0,0,0,0,21,102,102,183,233,212,81,0,0,0,0,0,0,0,0,0,0,0,0,82,255,253,234,152,153,193,173,253,254,253,254,213,142,20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,71,151,151,232,253,212,192,151,131,50,50,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

print("@@@")
print(Weights)
print("@@@")
print(bias)

# array_0 = []
# i = 0
# while i < 784:
#     array_0.append(Weights2[i][0])
#     i = i + 1

# my_number_array = []
# height = 0
# while height < 28:
#     line_array = []
#     width = 0
#     while width < 28:
#         line_array.append(array_0[(height * 28) + width])
#         width = width + 1
#     my_number_array.append(line_array)
#     height = height + 1

# data = np.asarray(my_number_array)
# #Rescale to 0-255 and convert to uint8
# rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
# im = Image.fromarray(rescaled)
# # im.save('test.png')



probability_0 = 0
i = 0
while i < 284:
    probability_0 = probability_0 + (Weights[i][0] * I[i])
    i = i+1

probability_1 = 0
i = 0
while i < 284:
    probability_1 = probability_1 + (Weights[i][1] * I[i])
    i = i+1

probability_2 = 0
i = 0
while i < 284:
    probability_2 = probability_2 + (Weights[i][2] * I[i])
    i = i+1

probability_3 = 0
i = 0
while i < 284:
    probability_3 = probability_3 + (Weights[i][3] * I[i])
    i = i+1

probability_4 = 0
i = 0
while i < 284:
    probability_4 = probability_4 + (Weights[i][4] * I[i])
    i = i+1

probability_5 = 0
i = 0
while i < 284:
    probability_5 = probability_5 + (Weights[i][5] * I[i])
    i = i+1

probability_6 = 0
i = 0
while i < 284:
    probability_6 = probability_6 + (Weights[i][6] * I[i])
    i = i+1

probability_7 = 0
i = 0
while i < 284:
    probability_7 = probability_7 + (Weights[i][7] * I[i])
    i = i+1

probability_8 = 0
i = 0
while i < 284:
    probability_8 = probability_8 + (Weights[i][8] * I[i])
    i = i+1

probability_9 = 0
i = 0
while i < 284:
    probability_9 = probability_9 + (Weights[i][9] * I[i])
    i = i+1

probability_array = [probability_0/255, probability_1/255, probability_2/255, probability_3/255, probability_4/255, probability_5/255, probability_6/255, probability_7/255, probability_8/255, probability_9/255]


# mean = np.mean(probability_array)
# std = np.std(probability_array)

final_array = []
i = 0
while i < 10:
    final_array.append(probability_array[i] + bias[i])
    i = i + 1

i = 0
while i < 10:
    print("%i:"%(i))
    print(final_array[i])
    i = i + 1

