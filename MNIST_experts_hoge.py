# -*- coding: utf-8 -*-

# TensowFlowのインポート
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import time

# 開始時刻
start_time = time.time()
print("開始時刻: " + str(start_time))
print( "--- MNISTデータの読み込み開始 ---")
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
print( "--- MNISTデータの読み込み完了 ---")
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

### 第一畳み込み層 ####################################################################

# 重みの初期化
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

### 第二畳み込み層 ####################################################################

# 重みの初期化
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

### 密に接続された層 ##################################################################

# 重みの初期化
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

### Dropout ########################################################################
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

### 読み出し層 #######################################################################
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交差エントロピーの計算
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

# 勾配硬化法を用い交差エントロピーが最小となるようyを最適化する
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.global_variables_initializer())

# 20000回の訓練（train_step）を実行する
print("--- 多層畳み込みネットワークによる訓練開始 ---")
for i in range(10):
  batch = mnist.train.next_batch(50)
  #print(np.shape(batch[0]))
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("--- 多層畳み込みネットワークによる訓練終了 ---")

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

# 終了時刻
end_time = time.time()
print( "終了時刻: " + str(end_time))
print( "かかった時間: " + str(end_time - start_time))

batch_x, batch_t = mnist.train.next_batch(1)
print('x_image',sess.run(tf.shape(x_image), feed_dict={x:batch_x, y_:batch_t, keep_prob:1.0}))
print('h_conv2',sess.run(tf.shape(h_conv2), feed_dict={x:batch_x, y_:batch_t, keep_prob:1.0}))
print('h_pool2',sess.run(tf.shape(h_pool2), feed_dict={x:batch_x, y_:batch_t, keep_prob:1.0}))
print('h_pool2_flat',sess.run(tf.shape(h_pool2_flat), feed_dict={x:batch_x, y_:batch_t, keep_prob:1.0}))
print('h_fc1',sess.run(tf.shape(h_fc1), feed_dict={x:batch_x, y_:batch_t, keep_prob:1.0}))
print('y_conv',sess.run(tf.shape(y_conv), feed_dict={x:batch_x, y_:batch_t, keep_prob:1.0}))

print('y_conv')
print(sess.run(y_conv, feed_dict={x:batch_x, y_:batch_t, keep_prob:1.0}))
