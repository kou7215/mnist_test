from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf
import time

start = time.time()

# MNISTは28*28=784次元. Noneは多くの入力データをぶちこむため次元が任意.
x = tf.placeholder(tf.float32, [None, 784])

# 入力次元784, 出力次元10
W = tf.Variable(tf.zeros([784, 10]))

# バイアス項. 10次元
b = tf.Variable(tf.zeros([10]))

# tf.matmul(x,W)でxWの行列の掛け算
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 正解ラベルのセット. 10次元に出力するので10次元
y_ = tf.placeholder(tf.float32, [None, 10])

# 損失関数の定義. 交差エントロピーを利用
# H(y) = - Σy_ * log(y)
#      = - Σ(正解データ) * log(実測データ)
# どうやらクロスエントロピーの平均を用いた場合は学習率を0.5にした方がよいが、
# 普通に使う場合は学習率0.01にする. -> 学習が遅くなる?? -> むしろ早かった. 平均を出す処理が入ったから
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

# 損失関数(交差エントロピー)を小さくする.
# 学習率を大きくすると, 学習は早いが, 雑. 小さいと丁寧だが, 遅い.
#train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 初期化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 1000回学習. ミニバッチサイズ100で学習. (100サンプルずつまとめて学習)
# バッチ学習は全てのサンプルを一度に入力し, 全てのサンプルに対する誤差(二乗誤差, 交差エントロピー)
# を計算する. 逐次(オンライン)学習は1つずつサンプルを入れて学習. ミニバッチと逐次学習は確率的勾配降下法であり, 
# 更新のたびにサンプルの傾向が変わるため, 局所解に陥りにくい.

# 訓練データは60000点あるので全て使いたいところだが費用つまり時間がかかるのでランダムな100つを使う
# 100つでも同じような結果を得ることができる
for i in range(1000):
    # ランダムに60000点から100サンプル抽出
    # どうやらbatch_xs, batch_ysは別のサンプルが抽出されるらしい
    batch_xs, batch_ys = mnist.train.next_batch(100)

    # feed_dictでplaceholderに値を入力することができる 
    # feeddictでxにbatch_xs,y_にbatch_ysを代入して、train_stepを実行している。
    # このときに、W,bの重みが更新される。
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# tf.argmax(y,1)で最も大きい出力をした(そのクラスに属する可能性の高い)インデックスを返す
# 訓練データと正解データで同じ値ならTrue.
# [True, False, True, True] は [1,0,1,1] になり、0.75
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# x, y_ にテストデータをぶち込んでみる.
# xに入力データ, y_に正解データ(ラベル)
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

timer = time.time() - start
print (("time:{0}".format(timer)) + "[sec]")
