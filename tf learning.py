import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 获取数据集
# 构建数学模型
x = tf.placeholder(tf.float32, [None, 784])# 该字符的特征
y_ = tf.placeholder(tf.float32, [None, 10])# 该字符的类别
W = tf.Variable(tf.zeros([784, 10])) # 权重矩阵
b = tf.Variable(tf.zeros([10])) # 偏置量
y = tf.nn.softmax(tf.matmul(x, W) + b) # 定义计算模型
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])) # 定义损失函数模型
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)# 梯度下降求最小损失
init = tf.global_variables_initializer() # 初始化图

with tf.Session() as sess:
    sess.run(init) # 初始化
    for i in range(1000): # 迭代训练1000次
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x:  batch_xs, y_: batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # 正确性检验
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 计算正确率
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))