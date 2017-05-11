import weights
from weights import Weights2
import numpy as np
import PIL
from PIL import Image
import tensorflow as tf

I = np.asarray(PIL.Image.open('images/its_a_zero.png'))
I = I.flatten()

a_four = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,218,253,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,217,252,92,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,200,252,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,204,250,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,79,249,149,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,109,252,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,207,252,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,126,252,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,160,252,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,229,252,253,103,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,53,184,255,171,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,253,228,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,253,228,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,253,228,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,253,228,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,213,253,235,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,201,253,252,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,253,252,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,253,252,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,64,195,241,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

w = Weights2

bias = [-0.12531213, 0.34291193, -0.07756195, -0.15483579, 0.14375827,  0.38140905,
 0.04379947, 0.30139488, -0.70874661, -0.14681587]

array_0 = []
i = 0
while i < 784:
    array_0.append(Weights2[i][0])
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



probability_0 = 0
i = 0
while i < 284:
    probability_0 = probability_0 + (w[i][0] * a_four[i])
    i = i+1

probability_1 = 0
i = 0
while i < 284:
    probability_1 = probability_1 + (w[i][1] * a_four[i])
    i = i+1

probability_2 = 0
i = 0
while i < 284:
    probability_2 = probability_2 + (w[i][2] * a_four[i])
    i = i+1

probability_3 = 0
i = 0
while i < 284:
    probability_3 = probability_3 + (w[i][3] * a_four[i])
    i = i+1

probability_4 = 0
i = 0
while i < 284:
    probability_4 = probability_4 + (w[i][4] * a_four[i])
    i = i+1

probability_5 = 0
i = 0
while i < 284:
    probability_5 = probability_5 + (w[i][5] * a_four[i])
    i = i+1

probability_6 = 0
i = 0
while i < 284:
    probability_6 = probability_6 + (w[i][6] * a_four[i])
    i = i+1

probability_7 = 0
i = 0
while i < 284:
    probability_7 = probability_7 + (w[i][7] * a_four[i])
    i = i+1

probability_8 = 0
i = 0
while i < 284:
    probability_8 = probability_8 + (w[i][8] * a_four[i])
    i = i+1

probability_9 = 0
i = 0
while i < 284:
    probability_9 = probability_9 + (w[i][9] * a_four[i])
    i = i+1

probability_array = [probability_0/255, probability_1/255, probability_2/255, probability_3/255, probability_4/255, probability_5/255, probability_6/255, probability_7/255, probability_8/255, probability_9/255]

mean = np.mean(probability_array)
std = np.std(probability_array)

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
