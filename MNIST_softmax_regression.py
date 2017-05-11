from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
import tensorflow as tf
import PIL
from PIL import Image

x = tf.placeholder(tf.float32, [None, 784])

#set weight and bias
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#implement model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#implement cross-entropy
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
  tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

#minimise the loss using gradient descent algo with a learning rate of 0.5
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#launch model in interactive session
sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for _ in range(5000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#does our prediction match the truth?
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

#get individual predictions. reduce image to pixels
I = np.asarray(PIL.Image.open('images/its_a_four.png').convert('L')) 
#needs black on the outside

print(len(I)) #need to convert I from array to dict
print(I[14]) #why is this 4 columns? RGB values?!
I = I.flatten()
print(len(I)) #why is the length 4x what it should be?! is this why it didn't
#work properly when multiplying by weights?!
#then, need to convert I from array to dict

a_four = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,50,224,0,0,0,0,0,0,0,70,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,121,231,0,0,0,0,0,0,0,148,168,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,195,231,0,0,0,0,0,0,0,96,210,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,69,252,134,0,0,0,0,0,0,0,114,252,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,236,217,12,0,0,0,0,0,0,0,192,252,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,247,53,0,0,0,0,0,0,0,18,255,253,21,0,0,0,0,0,0,0,0,0,0,0,0,0,84,242,211,0,0,0,0,0,0,0,0,141,253,189,5,0,0,0,0,0,0,0,0,0,0,0,0,0,169,252,106,0,0,0,0,0,0,0,32,232,250,66,0,0,0,0,0,0,0,0,0,0,0,0,0,15,225,252,0,0,0,0,0,0,0,0,134,252,211,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,252,164,0,0,0,0,0,0,0,0,169,252,167,0,0,0,0,0,0,0,0,0,0,0,0,0,0,9,204,209,18,0,0,0,0,0,0,22,253,253,107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,169,252,199,85,85,85,85,129,164,195,252,252,106,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,170,245,252,252,252,252,232,231,251,252,252,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,49,84,84,84,84,0,0,161,252,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127,252,252,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,128,253,253,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,127,252,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,135,252,244,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,232,236,111,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,179,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

# print (prediction.eval(feed_dict={x: mnist.test.images})[0])

prediction = tf.argmax(y,1)
a_four = np.reshape(a_four,(1, 784))
# print ("Prediction: %i"%prediction.eval(feed_dict={x: a_four}))
I = np.reshape(I,(1,784))
print ("Prediction: %i"%prediction.eval(feed_dict={x: I}))
