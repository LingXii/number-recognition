# 对将要识别的图片进行二值化、广搜并画出红框（以后要在此进行识别）
# 要求手写的数字连通，允许小噪点
import cv2
import queue
import numpy
from pylab import *
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True) # 获取数据集

# 机器学习模型
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
for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
# 使用测试集评估准确率
train_accuracy = accuracy.eval(feed_dict={
    x: batch_x, y_: batch_y, keep_prob: 0})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('test accuracy')
print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))


# image deal
q = queue.Queue() # 广搜队列
buf = queue.Queue() # 缓冲队列

def bfs(img,vis):
    x_size = img.shape[0]
    y_size = img.shape[1]
    first = q.get()
    up_border = first[0]
    left_border = first[1]
    dowm_border = first[0]
    right_border = first[1]
    q.put(first)
    while(not q.empty()):
        p = q.get()
        buf.put(p)  # 将广搜得到的像素点放入缓冲区，等待切分
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
    return (up_border,dowm_border,left_border,right_border)


origin_img = cv2.imread("2.jpg")
img = cv2.imread("2.jpg",0)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,75,25) # 二值化
img= cv2.medianBlur(img,3) # 去噪
x_size = img.shape[0]
y_size = img.shape[1]
visit = [[0 for y in range(y_size)] for x in range(x_size)]
cnt=0
for j in range(img.shape[1]):# 广搜
    for i in range(img.shape[0]):
        if (visit[i][j]==0 and img[i, j] < 30):
            visit[i][j]=1
            q.put((i,j));
            box = bfs(img,visit) # 一个框
            # 切分识别
            height = box[1]-box[0]+1
            width = box[3]-box[2]+1
            if(height*width < 300):
                while(not buf.empty()): buf.get()
                continue # 排除掉很小的区域，肯定是噪点
            if(height > width):
                bias = (height - width) >> 1
                simple_img = numpy.zeros((height,height))
                while(not buf.empty()):
                    p = buf.get()
                    simple_img[p[0]-box[0],p[1]-box[2]+bias] = 255-img[p[0],p[1]]
            else:
                bias = (width - height) >> 1
                simple_img = numpy.zeros((width, width))
                while (not buf.empty()):
                    p = buf.get()
                    simple_img[p[0]-box[0]+bias, p[1]-box[2]] = 255-img[p[0], p[1]]
            #机器学习模型尝试
            mnist_img = cv2.resize(simple_img, (28, 28))
            cv2.imwrite("./data/%s.jpg"%cnt,mnist_img)
            cnt=cnt+1
            for ii in range(28):
                for jj in range(28):
                    mnist_img[ii, jj] = mnist_img[ii, jj] / 255
            mnist_array = numpy.reshape(mnist_img, (1, 784))
            mnist_array = mnist_array.astype(numpy.float32)
            result = tf.arg_max(y,1)  # 计算数字概率
            prob = tf.nn.softmax(y)
            cal_y = sess.run(prob, feed_dict={x: mnist_array})
            cal_re = sess.run(result, feed_dict={x: mnist_array})
            if(cal_re[0]==5 or cal_re[0]==3): # 探查‘5’的笔画分离问题
                if (height > width):
                    for ii in range(height>>1):
                        for jj in range(width):
                            if(simple_img[ii,jj+bias] == img[ii+box[0],jj+box[2]] and visit[ii+box[0]][jj+box[2]]==0):
                                for iii in range(height):
                                    for jjj in range(width):
                                        if(simple_img[iii,jjj+bias] == 255): buf.put((iii+box[0],jjj+box[2]))
                                visit[ii+box[0]][jj+box[2]] = 1
                                q.put((ii+box[0], jj+box[2]));
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
                                else:
                                    bias = (width - height) >> 1
                                    simple_img = numpy.zeros((width, width))
                                    while (not buf.empty()):
                                        p = buf.get()
                                        simple_img[p[0] - box[0] + bias, p[1] - box[2]] = 255 - img[p[0], p[1]]
                else:
                    for ii in range(height>>1):
                        for jj in range(width):
                            if(simple_img[ii+bias,jj] == img[ii+box[0],jj+box[2]] and visit[ii+box[0]][jj+box[2]]==0):
                                for iii in range(height):
                                    for jjj in range(width):
                                        if(simple_img[iii+bias,jjj] == 255): buf.put((iii+box[0],jjj+box[2]))
                                visit[ii+box[0]][jj+box[2]] = 1
                                q.put((ii+box[0], jj+box[2]));
                                expand = bfs(img,visit) # 一个框
                                box0 = min(box[0], expand[0])
                                box1 = max(box[1], expand[1])
                                box2 = min(box[2], expand[2])
                                box3 = max(box[3], expand[3])
                                height = box[1] - box[0]+1
                                width = box[3] - box[2]+1
                                if (height > width):
                                    bias = (height - width) >> 1
                                    simple_img = numpy.zeros((height, height))
                                    while (not buf.empty()):
                                        p = buf.get()
                                        simple_img[p[0] - box[0], p[1] - box[2] + bias] = 255 - img[p[0], p[1]]
                                else:
                                    bias = (width - height) >> 1
                                    simple_img = numpy.zeros((width, width))
                                    while (not buf.empty()):
                                        p = buf.get()
                                        simple_img[p[0] - box[0] + bias, p[1] - box[2]] = 255 - img[p[0], p[1]]
                mnist_img = cv2.resize(simple_img, (28, 28))
                cv2.imwrite("./data/%s.jpg" % cnt, mnist_img)
                cnt = cnt + 1
                for ii in range(28):
                    for jj in range(28):
                        mnist_img[ii, jj] = mnist_img[ii, jj] / 255
                mnist_array = numpy.reshape(mnist_img, (1, 784))
                mnist_array = mnist_array.astype(numpy.float32)
                result = tf.arg_max(y, 1)  # 计算数字概率
                prob = tf.nn.softmax(y)
                cal_y = sess.run(prob, feed_dict={x: mnist_array})
                cal_re = sess.run(result, feed_dict={x: mnist_array})
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
cv2.imwrite("3.jpg",origin_img)
print("finish detect")