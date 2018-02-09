import tensorflow as tf

n1 = tf.constant(23)
n2 = tf.constant(12)

# 足し算tf.add()
add_op = tf.add(n1,n2)
with tf.Session() as sess:
    result = sess.run(add_op)
    print(result)
    print(sess.run(add_op))

# 引き算tf.subtract
subt_op = tf.subtract(n1,n2)
with tf.Session() as sess:
    print(sess.run(subt_op))
    print(n1-n2)    # cannot calc
    print(sess.run(n1-n2))    # can calc

# 割り算tf.divide()
div_op = tf.divide(n1,n2)
with tf.Session() as sess:
    print(sess.run(div_op))
    print("\n\n")
    print(sess.run(n1/n2))

# 内積・外積
a = tf.constant([1,2,3])
b = tf.constant([4,5,6])
# 内積
scalar_op = tf.scalar_mul(a,b)
# 外積
cross_op = tf.cross(a,b)
