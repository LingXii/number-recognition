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
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算正确率
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    result = tf.argmax(y,1) #计算样例图片识别结果
    print(sess.run(result,feed_dict={x:mnist_array}))

print("finish")