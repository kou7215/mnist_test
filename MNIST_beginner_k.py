import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

start = time.time()

# set value
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])
t = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros(10))
y = tf.nn.softmax(tf.matmul(x, W) + b)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1])) # 第二引数は平均をとる軸
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
# boolianなのでfloat32にキャストする必要ある.tfはキャスト必須だが, npならキャスト不要

# run
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    for i in range(1000):
        batch_x, batch_t = mnist.train.next_batch(100) # return (x, t)
        #sess.run(train_step, feed_dict={x:batch_x, t:batch_t})  # x,tにバッチを入れて訓練
        train_step.run({x:batch_x, t:batch_t})  # 単純にf.run({x:hoge, t:hoo})でも実行可能

    # sess.run()を使う方法とsess.runしないでeval()で対話的に呼び出す方法がある.
    # sess.run(f, feed_dict={x:hoge, t:hoo}) または f.eval({x:hoge, t:hoo})
    print("accuracy : ", sess.run(accuracy, feed_dict={x: mnist.test.images, t: mnist.test.labels}))    
    print("accuracy : ", accuracy.eval({x: mnist.test.images, t:mnist.test.labels}))    
    print("accuracy : ", accuracy.run({_mean(-tx: mnist.test.images, t:mnist.test.labels}))    

timer = time.time() - start
print(("time:{0}".format(timer)) + "[sec]")
