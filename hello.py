import tensorflow as tf
hello = tf.constant("Hello, TensorFlow!")
print (hello)           # print()で型表示
sess = tf.Session()     # sessionの定義. データフローグラフの計算に必要 
print(sess.run(hello))  # sess.run()で計算実行
a = tf.constant(10)
b = tf.constant(32)
print(sess.run(a+b))

# 足し算してみる
num1 = tf.constant(1)
num2 = tf.constant(2)
num3 = tf.constant(3)
num1PlusNum2 = tf.add(num1, num2)
num1PlusNum2PlusNum3 = tf.add(num1PlusNum2, num3)
sess = tf.Session()     # 最後のノードまで処理を書いてからセッションする
result = sess.run(num1PlusNum2PlusNum3)      # 実行
print(result)

