from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import numpy as np
import tensorflow as tf
import PIL
from PIL import Image
from Naked.toolshed.shell import execute_rb
success = execute_rb('app/models/photo.rb')

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
I = np.asarray(PIL.Image.open(photo_path).convert('L'))

print(len(I)) #need to convert I from array to dict
I = I.flatten()
print(len(I))

prediction = tf.argmax(y,1)
I = np.reshape(I,(1,784))
print ("Prediction: %i"%prediction.eval(feed_dict={x: I}))
