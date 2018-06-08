# 对将要识别的图片进行二值化、广搜并画出红框，对框内数字进行识别
# 背景作业本，要求手写的数字不粘连，允许噪点和污渍
# 训练+识别 程序运行时间约100分钟
# 用到了几个开源代码：https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/tree/master/tensorflow-mnist-tutorial
import cv2
import queue
import numpy
from pylab import *
from delete_line import *
import tensorflow as tf
import tensorflowvisu
import math
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 获取数据集
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)
tf.set_random_seed(0.0)

# 机器学习模型
# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
# # 加载数据
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# x = tf.placeholder(tf.float32, [None, 784])
# W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
# b1 = tf.Variable(tf.zeros([500]))
# W2 = tf.Variable(tf.truncated_normal([500, 10], stddev=0.1))
# b2 = tf.Variable(tf.zeros([10]))
# layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# y = tf.matmul(layer1, W2) + b2
#
# # 正确的样本标签
# y_ = tf.placeholder(tf.float32, [None, 10])
#
# # 损失函数选择softmax后的交叉熵，结果作为y的输出
# cross_entropy = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# sess = tf.InteractiveSession()
# tf.global_variables_initializer().run()
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# keep_prob = tf.placeholder(tf.float32)
# # 训练过程
# for i in range(10000):
#     batch_x, batch_y = mnist.train.next_batch(50)
#     sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
# # 使用测试集评估准确率
# train_accuracy = accuracy.eval(feed_dict={
#     x: batch_x, y_: batch_y, keep_prob: 0})
# correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print('test accuracy')
# print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                     y_: mnist.test.labels}))
# 老模型

# 新模型 from HR4.2
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

K = 24  # first convolutional layer output depth
L = 48  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6, 6, 1, K], stddev=0.1))  # 6x6 patch, 1 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

stride = 1  # output is 28x28
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1r = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))
stride = 2  # output is 14x14
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2r = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))
stride = 2  # output is 7x7
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3r = tf.nn.relu(Y3bn)
Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))
# reshape the output from the third convolution for the fully connected layer
YY = tf.reshape(Y3, shape=[-1, 7 * 7 * M])
Y4l = tf.matmul(YY, W4)
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, pkeep)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# matplotlib visualisation
allweights = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])], 0)
allbiases  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])], 0)
conv_activations = tf.concat([tf.reshape(tf.reduce_max(Y1r, [0]), [-1]), tf.reshape(tf.reduce_max(Y2r, [0]), [-1]), tf.reshape(tf.reduce_max(Y3r, [0]), [-1])], 0)
dense_activations = tf.reduce_max(Y4r, [0])
I = tensorflowvisu.tf_format_mnist_images(X, Y, Y_)
It = tensorflowvisu.tf_format_mnist_images(X, Y, Y_, 1000, lines=25)
datavis = tensorflowvisu.MnistDataVis(title4="batch-max conv activation", title5="batch-max dense activations", histogram4colornum=2, histogram5colornum=2)

# training step
# the learning rate is: # 0.0001 + 0.03 * (1/e)^(step/1000)), i.e. exponential decay from 0.03->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.02, iter, 1600, 1/math.e)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# You can call this function in a loop to train the model, 100 images at a time
def training_step(i, update_test_data, update_train_data):

    # training on batches of 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    # compute training values for visualisation
    if update_train_data:
        a, c, im, ca, da, l = sess.run([accuracy, cross_entropy, I, conv_activations, dense_activations, lr],
                                    feed_dict={X: batch_X, Y_: batch_Y, iter: i, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
        # print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c) + " (lr:" + str(l) + ")")
        datavis.append_training_curves_data(i, a, c)
        datavis.update_image1(im)
        datavis.append_data_histograms(i, ca, da)

    # compute test values for visualisation
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It],
                            feed_dict={X: mnist.test.images, Y_: mnist.test.labels, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
        # print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)

    # the backpropagation training step
    sess.run(train_step, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 0.75, pkeep_conv: 1.0})
    sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0,  pkeep_conv: 1.0})

# image deal
q = queue.Queue() # 广搜队列
buf = queue.Queue() # 缓冲队列

def bfs(img,vis):
    x_size = img.shape[0]
    y_size = 2060
    first = q.get()
    up_border = first[0]
    left_border = first[1]
    dowm_border = first[0]
    right_border = first[1]
    count=0
    q.put(first)
    while(not q.empty()):
        p = q.get()
        buf.put(p)
        count=count+1
        if(p[0]>dowm_border): dowm_border=p[0]
        if(p[1]>right_border): right_border=p[1]
        if(p[0]<up_border):
            up_border=p[0]
            first = (p[0],first[1])
        if(p[1]<left_border):
            left_border=p[1]
            first = (first[0],p[1])
        left = (p[0] - 1, p[1])
        right = (p[0] + 1, p[1])
        up = (p[0], p[1] - 1)
        down = (p[0], p[1] + 1)
        if (left[0] >= 0 and vis[left[0]][left[1]] == 0 and img[left[0],left[1]]<30):
            q.put(left)
            vis[left[0]][left[1]] = 1
        if (right[0] < x_size and vis[right[0]][right[1]] == 0 and img[right[0],right[1]]<30):
            q.put(right)
            vis[right[0]][right[1]] = 1
        if (up[1] >= 0 and vis[up[0]][up[1]] == 0 and img[up[0],up[1]]<30):
            q.put(up)
            vis[up[0]][up[1]] = 1
        if (down[1] < y_size and vis[down[0]][down[1]] == 0 and img[down[0],down[1]]<30):
            q.put(down)
            vis[down[0]][down[1]] = 1
        if (left[0] >= 0 and vis[left[0]][left[1]] == 0 and img[left[0],left[1]]==128):
            q.put(left)
            vis[left[0]][left[1]] = 1
        if (right[0] < x_size and vis[right[0]][right[1]] == 0 and img[right[0],right[1]]==128):
            q.put(right)
            vis[right[0]][right[1]] = 1
    return (up_border,dowm_border,left_border,right_border,count)

kernel = numpy.uint8(numpy.zeros((5, 5)))
for x in range(5):
    kernel[x, 2] = 1;
    kernel[2, x] = 1;
model_saver = tf.train.Saver() #模型的储存
# for i in range(10001):
#     if (i==0): continue
#     if(i==10000): model_saver.save(sess,"./model/num_rec")
#     if(i%100==0): print("training"+str(i//100)+"%")
#     training_step(i,0,1)
model_saver.restore(sess, "./model/saved_model4.ckpt")
print("finish training")
origin_img = cv2.imread("T9.jpg")
img = cv2.imread("T9.jpg",0)
img = del_line(img)
x_size = img.shape[0]
y_size = 2060
visit = [[0 for y in range(y_size)] for x in range(x_size)]
cnt=0
for j in range(425,2060):# 广搜
    for i in range(550,img.shape[0]):
        if (visit[i][j]==0 and img[i, j] < 30):
            visit[i][j]=1
            q.put((i,j))
            box = bfs(img,visit) # 一个框
            # 切分识别
            height = box[1]-box[0]+1
            width = box[3]-box[2]+1
            if(height*width < 550):
                while(not buf.empty()): buf.get()
                continue # 排除掉很小的区域，肯定是噪点
            if(width>height/3 and box[4]>(height*width)/5*3):
                while (not buf.empty()): buf.get()
                continue  # 排除掉黑点占比很大的区域，肯定是污渍
            if(height > width):
                bias = (height - width) >> 1
                simple_img = numpy.zeros((height,height))
                while(not buf.empty()):
                    p = buf.get()
                    simple_img[p[0]-box[0],p[1]-box[2]+bias] = 255
            else:
                bias = (width - height) >> 1
                simple_img = numpy.zeros((width, width))
                while (not buf.empty()):
                    p = buf.get()
                    simple_img[p[0]-box[0]+bias, p[1]-box[2]] = 255
            #机器学习模型尝试
            if (simple_img.shape[0] > 300):
                simple_img2 = cv2.dilate(simple_img, kernel)
                simple_img2 = cv2.dilate(simple_img2, kernel)
                simple_img2 = cv2.dilate(simple_img2, kernel)
            elif (simple_img.shape[0] > 200):
                simple_img2 = cv2.dilate(simple_img, kernel)
                simple_img2 = cv2.dilate(simple_img2, kernel)
            elif (simple_img.shape[0] > 100):
                simple_img2 = cv2.dilate(simple_img, kernel)
            else:
                simple_img2 = simple_img.copy()
            mnist_img = cv2.resize(simple_img2, (28, 28))
            cv2.imwrite("./data/%s.jpg" % cnt, mnist_img)
            cnt = cnt + 1
            for ii in range(28):
                for jj in range(28):
                    mnist_img[ii, jj] = mnist_img[ii, jj] / 255
            mnist_array = numpy.reshape(mnist_img, (1,28,28,1))
            mnist_array = mnist_array.astype(numpy.float32)
            result = tf.arg_max(Y,1)  # 计算数字概率
            prob = tf.nn.softmax(Y)
            cal_y = sess.run(prob, feed_dict={X: mnist_array, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
            cal_re = sess.run(result, feed_dict={X: mnist_array, tst: True, pkeep: 1.0, pkeep_conv: 1.0})

            if(cal_re[0]==5 or cal_re[0]==3): # 探查‘5’的笔画分离问题
                if (height > width):
                    b=0
                    for ii in range(height>>1):
                        for jj in range(width):
                            if(simple_img[ii,jj+bias] == img[ii+box[0],jj+box[2]] and visit[ii+box[0]][jj+box[2]]==0):
                                for iii in range(height):
                                    for jjj in range(width):
                                        if(simple_img[iii,jjj+bias] == 255): buf.put((iii+box[0],jjj+box[2]))
                                visit[ii+box[0]][jj+box[2]] = 1
                                q.put((ii+box[0], jj+box[2]))
                                expand = bfs(img,visit) # 一个框
                                box0 = min(box[0],expand[0])
                                box1 = max(box[1],expand[1])
                                box2 = min(box[2],expand[2])
                                box3 = max(box[3],expand[3])
                                box = (box0,box1,box2,box3)
                                height = box[1] - box[0]+1
                                width = box[3] - box[2]+1
                                if (height > width):
                                    bias = (height - width) >> 1
                                    simple_img = numpy.zeros((height, height))
                                    while (not buf.empty()):
                                        p = buf.get()
                                        simple_img[p[0] - box[0], p[1] - box[2] + bias] = 255 - img[p[0], p[1]]
                                    b=1
                                else:
                                    bias = (width - height) >> 1
                                    simple_img = numpy.zeros((width, width))
                                    while (not buf.empty()):
                                        p = buf.get()
                                        simple_img[p[0] - box[0] + bias, p[1] - box[2]] = 255 - img[p[0], p[1]]
                                    b=1
                            if(b==1): break
                        if(b==1): break
                else:
                    b = 0
                    for ii in range(height>>1):
                        for jj in range(width):
                            if(simple_img[ii+bias,jj] == img[ii+box[0],jj+box[2]] and visit[ii+box[0]][jj+box[2]]==0):
                                for iii in range(height):
                                    for jjj in range(width):
                                        if(simple_img[iii+bias,jjj] == 255): buf.put((iii+box[0],jjj+box[2]))
                                visit[ii+box[0]][jj+box[2]] = 1
                                q.put((ii+box[0], jj+box[2]))
                                expand = bfs(img,visit) # 一个框
                                box0 = min(box[0], expand[0])
                                box1 = max(box[1], expand[1])
                                box2 = min(box[2], expand[2])
                                box3 = max(box[3], expand[3])
                                box = (box0, box1, box2, box3)
                                height = box[1] - box[0]+1
                                width = box[3] - box[2]+1
                                if (height > width):
                                    bias = (height - width) >> 1
                                    simple_img = numpy.zeros((height, height))
                                    while (not buf.empty()):
                                        p = buf.get()
                                        simple_img[p[0] - box[0], p[1] - box[2] + bias] = 255 - img[p[0], p[1]]
                                    b = 1
                                else:
                                    bias = (width - height) >> 1
                                    simple_img = numpy.zeros((width, width))
                                    while (not buf.empty()):
                                        p = buf.get()
                                        simple_img[p[0] - box[0] + bias, p[1] - box[2]] = 255 - img[p[0], p[1]]
                                    b = 1
                            if (b == 1): break
                        if (b == 1): break
                if (simple_img.shape[0] > 300):
                    simple_img2 = cv2.dilate(simple_img, kernel)
                    simple_img2 = cv2.dilate(simple_img2, kernel)
                    simple_img2 = cv2.dilate(simple_img2, kernel)
                elif (simple_img.shape[0] > 200):
                    simple_img2 = cv2.dilate(simple_img, kernel)
                    simple_img2 = cv2.dilate(simple_img2, kernel)
                elif (simple_img.shape[0] > 100):
                    simple_img2 = cv2.dilate(simple_img, kernel)
                else:
                    simple_img2 = simple_img.copy()
                mnist_img = cv2.resize(simple_img2, (28, 28))
                for ii in range(28):
                    for jj in range(28):
                        mnist_img[ii, jj] = mnist_img[ii, jj] / 255
                mnist_array = numpy.reshape(mnist_img, (1, 28, 28, 1))
                mnist_array = mnist_array.astype(numpy.float32)
                result = tf.arg_max(Y, 1)  # 计算数字概率
                prob = tf.nn.softmax(Y)
                cal_y = sess.run(prob, feed_dict={X: mnist_array, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
                cal_re = sess.run(result, feed_dict={X: mnist_array, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
            # 画边框
            boxj = box[2]
            while (boxj <= box[3]):
                origin_img[box[0], boxj] = (0, 0, 255)
                origin_img[box[1], boxj] = (0, 0, 255)
                origin_img[box[0] + 1, boxj] = (0, 0, 255)
                origin_img[box[1] - 1, boxj] = (0, 0, 255)
                boxj = boxj + 1
            boxi = box[0]
            while (boxi <= box[1]):
                origin_img[boxi, box[2]] = (0, 0, 255)
                origin_img[boxi, box[3]] = (0, 0, 255)
                origin_img[boxi, box[2] + 1] = (0, 0, 255)
                origin_img[boxi, box[3] - 1] = (0, 0, 255)
                boxi = boxi + 1
            #写数字
            str = "%s:"%cal_re[0]
            str = str + "%.5f"%cal_y[0][cal_re[0]]
            font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
            origin_img = cv2.putText(origin_img, str, (box[3], box[1]), font, 0.8, 0, 2)
cv2.imwrite("7.jpg",origin_img)
print("finish detect")