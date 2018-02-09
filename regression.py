import tensorflow as tf
import numpy as np
import time

# 計測開始
start = time.time()

# 実データ
# y = 0.1x + 0.3に乗る点を乱数で100個生成(float32型)
x_data = np.random.rand(100).astype("float32") # 0.0~1.0の乱数を100個生成
y_data = 0.1 * x_data + 0.3

# 訓練データの生成
# tf.random_uniform([要素数], n1, n2)
# n1 ~ n2の範囲で[要素数]の乱数生成. この場合1個
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# 1次元の要素を持つテンソルをすべて0で初期化
b = tf.Variable(tf.zeros([1]))
# 乱数による線形回帰モデル(仮説関数)
y = W * x_data + b

# コスト関数の定義
# 二乗平均誤差 1/2(y_data - y)^2
# reduce_meanと書いてあるが, ただの平均
loss = tf.reduce_mean(tf.square(y_data - y))

# 勾配降下法. 引数は学習率(0.5)
# 誤差関数を微分して学習率をかけたものを引いて更新. 勾配0（最小値）になるまで微分して引き最適化
train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


#init = tf.initialize_all_variables()   <- 古いver
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(1001):
    sess.run(train)
    if step % 100 == 0:
        print (step, sess.run(W), sess.run(b))

for i in range(100):
    print (i, sess.run(W) * i + sess.run(b))

sess.close()

timer = time.time() - start
print (("time:{0}".format(timer)) + "[sec]")
