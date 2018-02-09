#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


import tensorflow as tf

# Import data
#import input_data
#mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# Variables
# 訓練データを格納するプレースホルダー
x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

# μ=0.0, σ=0.05 の正規分布から乱数生成. 隠れ層, 出力層のパラメータ
# 中間層の次元を小さくしても大して変わらんっぽい.
w_h = tf.Variable(tf.random_normal([784, 625], mean=0.0, stddev=0.05))
w_o = tf.Variable(tf.random_normal([625, 10], mean=0.0, stddev=0.05))
b_h = tf.Variable(tf.zeros([625]))
b_o = tf.Variable(tf.zeros([10]))

# Create the model
def model(X, w_h, b_h, w_o, b_o):
    # 隠れ層はシグモイド関数, 出力層はソフトマックス
    h = tf.sigmoid(tf.matmul(X, w_h) + b_h)
    pyx = tf.nn.softmax(tf.matmul(h, w_o) + b_o)

    return pyx  # 最終的に出力層のソフトマックスのみ返せば良い

# 最終出力データ
y_hypo = model(x, w_h, b_h, w_o, b_o)

# Cost Function basic term
# コスト関数として交差エントロピーを利用
# H(y) = - Σy_ * log(y)
#      = - Σ(正解データ)*log(実測データ)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_hypo))

# Regularization terms (weight decay)
# リッジ回帰による正則化項. λ * (w_h)^2 + (w_o)^2
# tf.nn.l2_loss()で二乗誤差を計算
L2_sqr = tf.nn.l2_loss(w_h) + tf.nn.l2_loss(w_o)
lambda_2 = 0.01

# the loss and accuracy
loss = cross_entropy + lambda_2 * L2_sqr
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y_hypo,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Train
#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print('Training...')
    for i in range(20001):
        batch_xs, batch_ys = mnist.train.next_batch(100)    # やはりミニバッチ学習
        train_step.run({x: batch_xs, y_: batch_ys})
        if i % 2000 == 0:
            train_accuracy = accuracy.eval({x: batch_xs, y_: batch_ys})
            print('  step, accurary = %6d: %6.3f' % (i, train_accuracy))

    # Test trained model
    print('accuracy = ', accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

