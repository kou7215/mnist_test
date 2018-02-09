import tensorflow as tf
sess = tf.Session()
batch_size = 3
output_shape = [batch_size, 8, 8, 128]
strides = [1, 2, 2, 1]


x = tf.constant(0.1, shape=[3,8,8,128])
w = tf.constant(0.1, shape=[7,7,128,4])
w2 = tf.constant(0.1, shape=w.shape)
conv1 = tf.nn.conv2d(x, w, strides=strides, padding='SAME')
deconv1 = tf.nn.conv2d_transpose(conv1, w2, output_shape=x.shape, strides=strides, padding='SAME')
# deconvのoutput_shapeは入力サイズと同じでOK(x.shape).
#wは転置せずにforwardと同じ形状のwを用いる

print(sess.run(x).shape)
print(sess.run(conv1).shape)
print(sess.run(deconv1).shape)
print(sess.run(w).shape)

