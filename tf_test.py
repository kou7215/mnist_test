import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
t = tf.placeholder(tf.float32, shape=[None, 10])
def Init_weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

def Init_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

def Conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def Max_Pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

################### ネットワーク構築  ######################################

W_conv1 = Init_weight([10,10,1,32])
b_conv1 = Init_bias([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = Conv2d(x_image, W_conv1)
y_1     = tf.nn.relu(h_conv1 + b_conv1)
h_pool1 = Max_Pooling_2x2(y_1)

W_conv2 = Init_weight([5,5,32,64])    # 入力32, 出力64
b_conv2 = Init_bias([64])
h_conv2 = tf.nn.relu(Conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = Max_Pooling_2x2(h_conv2)
#
## 全結合
## h_pool2で出力されたテンソルを2次元のベクトル(行列)に整形する
#W_fc1 = Init_weight([7*7*64, 1024])   # 出力1024は任意
#b_fc1  = Init_bias([1024])
#h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
#


sess.run(tf.global_variables_initializer())
x_batch, t_batch = mnist.train.next_batch(10)
print("x_image ", sess.run(tf.shape(x_image), feed_dict={x:x_batch}))
print("h_conv1 ", sess.run(tf.shape(h_conv1), feed_dict={x:x_batch}))
print("y_1 ", sess.run(tf.shape(y_1), feed_dict={x:x_batch}))
print("h_pool1 ", sess.run(tf.shape(h_pool1), feed_dict={x:x_batch}))

print("h_conv2 ", sess.run(tf.shape(h_conv2), feed_dict={x:x_batch}))
print("h_pool2 ", sess.run(tf.shape(h_pool2), feed_dict={x:x_batch}))

