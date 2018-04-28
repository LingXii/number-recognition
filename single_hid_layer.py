
#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
# 加载数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
"""
# 创建模型
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b
"""
x = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]))
W2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
b2 = tf.Variable(tf.zeros([10]))
layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
y = tf.matmul(layer1, W2) + b2


# 正确的样本标签
y_ = tf.placeholder(tf.float32, [None, 10])

# 损失函数选择softmax后的交叉熵，结果作为y的输出
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
keep_prob = tf.placeholder(tf.float32)
# 训练过程
for i in range(20000):
    if i<18000:
        batch_x, batch_y = mnist.train.next_batch(50)
    if i>=18000:
        batch_x, batch_y = mnist.train.next_batch(3000)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    if i%1000 == 0:
        # 使用测试集评估准确率
        train_accuracy = accuracy.eval(feed_dict={
                x:batch_x, y_: batch_y, keep_prob: 0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
        
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('test accuracy')
        print (sess.run(accuracy, feed_dict = {x: mnist.test.images,
                                                  y_: mnist.test.labels}))