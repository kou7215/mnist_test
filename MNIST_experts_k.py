import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Session()の場合, グラフを構築してからSession()しないといけないが
# InteractiveSession()の場合は計算グラフを作りながらSessionを挟める
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
t = tf.placeholder(tf.float32, shape=[None, 10])
# 重みの初期化. 切断正規分布を利用
def Init_weight(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)

# バイアスの初期化(0.1)
def Init_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)

# 畳み込み. strides[0], [3]は1で固定
def Conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

# プーリング(2 X 2). ksize:フィルタサイズ(2x2)
def Max_Pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

################### ネットワーク構築  ######################################

# 1層 
# weight([フィルタ縦, フィルタ横, チャネル, 特徴量(出力数)])
# 5x5パッチサイズ, 1チャンネル(白黒), 出力32特徴量
W_conv1 = Init_weight([5,5,1,32])   
b_conv1 = Init_bias([32])
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1 = tf.nn.relu(Conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = Max_Pooling_2x2(h_conv1)

# 2層
W_conv2 = Init_weight([5,5,32,64])    # 入力32, 出力64
b_conv2 = Init_bias([64])
h_conv2 = tf.nn.relu(Conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = Max_Pooling_2x2(h_conv2)

# 全結合
# h_pool2で出力されたテンソルを2次元のベクトル(行列)に整形する
W_fc1 = Init_weight([7*7*64, 1024])   # 出力1024は任意
b_fc1  = Init_bias([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# DropOut
# placeholderに格納することで, 訓練時にON, テスト時にOFFできる.
# 全結合のReLuの出力に対してDropOut
keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 読み出し層. 10クラス分類するためにsoftmaxへ落とし込む
W_fc2 = Init_weight([1024, 10])
b_fc2 = Init_bias([10])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2))

# クロスエントロピー/Adam/Accuracy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(t*tf.log(y), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 学習モデルの保存
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('/Users/konosuke-a/Desktop/mnist_test/hoge')  # 'checkpoint'が存在すればTrue
if ckpt:
    last_model = ckpt.model_checkpoint_path
    print("load" + last_model)
    saver.restore(sess, last_model)
else:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        x_batch, t_batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:x_batch, t:t_batch, keep_prob:1.0})
            print("step:%d   training accuracy:%g"%(i, train_accuracy))
        train_step.run(feed_dict={x:x_batch, t:t_batch, keep_prob:0.5})

test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images, t:mnist.test.labels, keep_prob:1.0})
print("test accuracy %g"%test_accuracy)

# 重み, バイアスの保存
saver.save(sess, "model.ckpt")
