import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
import numpy
from image_resize import *

# test recognition
img=cv2.imread("0.jpg",0)# 读入灰度图片
img = img_resize(img)
mnist_img = numpy.zeros((28,28))
for i in range(28):
    for j in range(28):
        mnist_img[i,j] = img[i,j]/255
mnist_array = numpy.reshape(mnist_img,(1,784))
mnist_array = mnist_array.astype(numpy.float32)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 获取数据集

# 构建机器学习模型（请加油改进此部分以提高识别准确率）
# 简单模型
# x = tf.placeholder(tf.float32, [None, 784])# 该字符的特征
# y_ = tf.placeholder(tf.float32, [None, 10])# 该字符的类别
# W = tf.Variable(tf.zeros([784, 10])) # 权重矩阵
# b = tf.Variable(tf.zeros([10])) # 偏置量
# y = tf.nn.softmax(tf.matmul(x, W) + b) # 定义计算模型
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1])) # 定义损失函数模型
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)# 梯度下降求最小损失
# init = tf.global_variables_initializer() # 初始化图
#
# with tf.Session() as sess:
#     sess.run(init) # 初始化
#     for i in range(1000): # 迭代训练1000次
#         batch_xs, batch_ys = mnist.train.next_batch(100)
#         sess.run(train_step, feed_dict={x:  batch_xs, y_: batch_ys})
#     correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # 正确性检验
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算正确率
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
#     result = tf.argmax(y,1) #计算样例图片识别结果
#     print(sess.run(result,feed_dict={x:mnist_array}))

# 升级模型
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 加载数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
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
for i in range(5000):
    batch_x, batch_y = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
# 使用测试集评估准确率
train_accuracy = accuracy.eval(feed_dict={
    x: batch_x, y_: batch_y, keep_prob: 0})
print("step %d, training accuracy %g" % (i, train_accuracy))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('test accuracy')
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
result = tf.argmax(y, 1)  # 计算样例图片识别结果
print(sess.run(result, feed_dict={x: mnist_array}))

print("finish")