import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=False)

H =2
BATCH_SIZE = 100
DROP_OUT_RATE = 0.5

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# input x : 28x28
x = tf.placeholder(tf.float32, [None, 784])
W = weight_variable((784, H))
b1 = bias_variable([H])

# Hidden Layer
h = tf.nn.softsign(tf.matmul(x, W) + b1)
keep_prob = tf.placeholder("float")
h_drop = tf.nn.dropout(h, keep_prob)
#h_drop = tf.nn.dropout(tf.nn.softsign(tf.matmul(x, W)+b1), keep_prob)

# output
W2 = tf.transpose(W) # shape(W) = (784, 50) -> shape(W2) = (50, 784)
b2 = bias_variable([784])
y = tf.nn.relu(tf.matmul(h_drop, W2) + b2)

# loss function and using tensorboard
loss = tf.reduce_mean(tf.square(y - x))
tf.summary.scalar("mean_square", loss)

# train step
train_step = tf.train.AdamOptimizer().minimize(loss)

# prepare Session
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter("./summary/mean_square", graph=sess.graph)

# training
for step in range(1001):
    batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    sess.run(train_step, feed_dict={x:batch_x, keep_prob:DROP_OUT_RATE})
    
    # correct summary
    summary_op = tf.summary.merge_all()
    summary_str = sess.run(summary_op, feed_dict={x:batch_x, keep_prob:1.0})
    summary_writer.add_summary(summary_str, step)
    # print progress
    if step % 100 == 0:
        print("step : %g, "%step, loss.eval(session=sess, feed_dict={x:batch_x, keep_prob:1.0}))

batch_x, _ = mnist.train.next_batch(1)
y_relu = y.eval(session=sess, feed_dict={x: batch_x, keep_prob:1.0})
# matplotlib -> なぜかカラー表示になる. たぶん0~1の小数値を用いてるからヒートマップで表示される
#plt.subplot(1,2,1)
#plt.imshow(batch_x.reshape(28,28))
#plt.subplot(1,2,2)
#plt.imshow(y_relu.reshape(28,28))
#plt.show()

## Pillow -> 白黒で出力できた. PILで画像化してpltで出力
#im_x = Image.fromarray(batch_x.reshape(28,28)*255)
#im_y = Image.fromarray(y_relu.reshape(28,28)*255)
#plt.subplot(1,2,1)
#plt.imshow(im_x)
#plt.subplot(1,2,2)
#plt.imshow(im_y)
#plt.show()

# AE plot 784dim -> 2dim
NUM_PLOT = 10000
batch_x, batch_t = mnist.train.next_batch(NUM_PLOT)
y_2d = h.eval(session=sess, feed_dict={x:batch_x})
#plot 0 ~ 9
plt.figure()
plt.title("Plot 2D all MNIST with Autoencoder")
for i in range(NUM_PLOT):
    if batch_t[i] == 0:
        p0, = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="red")
    if batch_t[i] == 1:
        p1, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="blue")
    if batch_t[i] == 2:
        p2, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="green")
    if batch_t[i] == 3:
        p3, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="pink")
    if batch_t[i] == 4:
        p4, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="yellow")
    if batch_t[i] == 5:
        p5, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="orange")
    if batch_t[i] == 6:
        p6, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="cyan")
    if batch_t[i] == 7:
        p7, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="deeppink")
    if batch_t[i] == 8:
        p8, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="c")
    if batch_t[i] == 9:
        p9, = plt.plot(y_2d[i,0], y_2d[i, 1], ".", c="purple")
plt.legend([p0,p1,p2,p3,p4,p5,p6,p7,p8,p9],["0","1","2","3","4","5","6","7","8","9"], bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_alldata_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 0 with Autoencoder")
for i in range(100):
    if batch_t[i] == 0:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="red")
plt.legend(p0,"0", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_0_2d.png"
plt.savefig(filename)


plt.figure()
plt.title("Plot 2D digits 1 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 1:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="blue")
plt.legend(p0,"1", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_1_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 2 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 2:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="green")
plt.legend(p0,"2", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_2_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 3 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 3:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="pink")
plt.legend(p0,"3", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_3_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 4 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 4:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="yellow")
plt.legend(p0,"4", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_4_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 5 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 5:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="orange")
plt.legend(p0,"5", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_5_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 6 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 6:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="cyan")
plt.legend(p0,"6", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_6_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 7 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 7:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="deeppink")
plt.legend(p0,"7", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_7_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 8 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 8:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="c")
plt.legend(p0,"8", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_8_2d.png"
plt.savefig(filename)

plt.figure()
plt.title("Plot 2D digits 9 with Autoencoder")
for i in range(10000):
    if batch_t[i] == 9:
        p0 = plt.plot(y_2d[i,0], y_2d[i, 1], ".",c="purple")
plt.legend(p0,"9", bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0)
filename = "AE_9_2d.png"
plt.savefig(filename)


